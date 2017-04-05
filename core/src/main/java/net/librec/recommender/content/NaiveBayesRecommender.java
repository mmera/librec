package net.librec.recommender.content;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.google.common.collect.*;
import net.librec.common.LibrecException;
import net.librec.math.structure.*;
import net.librec.recommender.AbstractRecommender;

public class NaiveBayesRecommender extends AbstractRecommender
{
	private static final int BSIZE = 1024 * 1024;
	private static final double alpha = 0.01;
	private static final int LIKE = 0;
	private static final int NOT_LIKE = 1;
	private static final int NOT_SEEN = -1;
	

	protected SparseMatrix m_featureMatrix;
	protected double ratingThreshold;
	private ArrayList<UserRatingProfile> userProfiles; // ArrayList to keep track of user profiles 
	/**
	 * Set up code expects a file name under the parameter dfs.content.path. It loads
	 * comma-separated feature data and stores a row at a time in the contentTable.
	 * After the data is loaded, it is converted into a SparseMatrix and stored in
	 * m_featureMatrix. This code borrows heavily from the implementation in
	 * net.librec.data.convertor.TextDataConvertor.
	 **/
	@Override
	public void setup() throws LibrecException {
		super.setup();
		String contentPath = conf.get("dfs.content.path");
		ratingThreshold = conf.getDouble("rec.rating.threshold");
		Table<Integer, Integer, Integer> contentTable = HashBasedTable.create();
		HashBiMap<String, Integer> itemIds = HashBiMap.create();
		HashBiMap<String, Integer> featureIds = HashBiMap.create();
		userProfiles = new ArrayList<>();

		try {
			FileInputStream fileInputStream = new FileInputStream(contentPath);
			FileChannel fileRead = fileInputStream.getChannel();
			ByteBuffer buffer = ByteBuffer.allocate(BSIZE);
			int len;
			String bufferLine = new String();
			byte[] bytes = new byte[BSIZE];
			while ((len = fileRead.read(buffer)) != -1) {
				buffer.flip();
				buffer.get(bytes, 0, len);
				bufferLine = bufferLine.concat(new String(bytes, 0, len));
				String[] bufferData = bufferLine.split(System.getProperty("line.separator") + "+");
				boolean isComplete = bufferLine.endsWith(System.getProperty("line.separator"));
				int loopLength = isComplete ? bufferData.length : bufferData.length - 1;
				for (int i = 0; i < loopLength; i++) {
					String line = new String(bufferData[i]);
					String[] data = line.trim().split("[ \t,]+");

					String item = data[0];
					int row = itemIds.containsKey(item) ? itemIds.get(item) : itemIds.size();
					itemIds.put(item, row);

					for (int j = 1; j < data.length; j++) {
						String feature = data[j];
						int col = featureIds.containsKey(feature) ? featureIds.get(feature) : featureIds.size();
						featureIds.put(feature, col);

						contentTable.put(row, col, 1);
					}

				}
			}
			fileInputStream.close();
		} catch (IOException e) {
			LOG.error("Error reading file: " + contentPath + e);
			throw (new LibrecException(e));
		}

		m_featureMatrix = new SparseMatrix(itemIds.size(), featureIds.size(), contentTable);
		LOG.info("Loaded item features from " + contentPath);
		
	}
	

	@Override
	public void trainModel() throws LibrecException {

		for(int userIdx=0; userIdx<numUsers; userIdx++){
			SparseVector userRatingVector = trainMatrix.row(userIdx); //Ratings of all rated items for a given user.
			UserRatingProfile user = new UserRatingProfile(userRatingVector);
			user.train();
			userProfiles.add(user); //This adds a given user and his propensity to like certain features to a collection of user profiles. 
		}
	}


	@Override
	public double predict(int userIdx, int itemIdx) throws LibrecException {

		UserRatingProfile profile = userProfiles.get(userIdx);
		List<Integer> itemFeatures = m_featureMatrix.row(itemIdx).getIndexList(); //Gets all the features of the item we are predicting on
		
		double pLike = profile.pLike;
		double pNotLike = profile.pNotLike;
		double productLike = 1;
		double productNotLike = 1;
		
		
		for(int feature:itemFeatures){
			/*
			 * If the feature has not been seen by the user, then we use the LaPlace variable i.e.
			 * [alpha / ([# of Items Liked] + (2 * alpha))] 
			 * 				 or 
			 * [alpha / ([# of Items Not Liked] + (2 * alpha))]
			 */
			if(!profile.userFeatureProbabilities.containsKey(feature)){ 
				productLike *= profile.userFeatureProbabilities.get(NOT_SEEN)[LIKE];
				productNotLike *= profile.userFeatureProbabilities.get(NOT_SEEN)[NOT_LIKE];
			}else{
				productLike *= profile.userFeatureProbabilities.get(feature)[LIKE];
				productNotLike *= profile.userFeatureProbabilities.get(feature)[NOT_LIKE];
			}
		}
		double k = pLike * productLike + pNotLike * productNotLike;
		double pY = (pLike * productLike) / k; // P(L | [Feature Set])
		double pN = (pNotLike * productNotLike) / k; //P(~L | [Feature Set])
		double logit = Math.log(pY) - Math.log(pN);
		double finalRatingPrediction = (1 / (1 + Math.exp(-logit))) * 4 + 1; //Inverse logit to get probability and adjust for range of ratings (1-5).
		
		return finalRatingPrediction;

	}
	/*
	 * A helper class to keep track of information between trainModel() and predict() for a given user. 
	 */
	private class UserRatingProfile{
		double numLiked;//    Total items liked by user
		double numNotLiked;// Total items not liked by user 
		double numRated;//    Total items rated
		double pLike; //		P(L)
		double pNotLike; //	    P(~L)
		HashMap<Integer,double[]> userFeatureProbabilities; // {feature i: [P(L | i),P(~L | i)] }
		HashMap<Integer,double[]> featureCount; // {feature i: [# of times feature appears in all liked items, # of times feature appears in all items not liked] }
		SparseVector itemsRatings;
		List<Integer> ratedItems;
		
		/*
		 * Constructor takes a sparse vector of ratings and computes things like numLiked, numRated and associated probabilities 
		 */
		UserRatingProfile(SparseVector itemsRatings){
			this.itemsRatings = itemsRatings;
			this.ratedItems = itemsRatings.getIndexList();
			for(int item:ratedItems){
				if(itemsRatings.get(item) >= ratingThreshold){
					numLiked++;
				}
			}
			numRated = ratedItems.size();
			numNotLiked = numRated - numLiked;
			pLike = (numLiked + alpha) / (numRated + 2 * alpha); 
			pNotLike = (numNotLiked + alpha) / (numRated + 2 * alpha);
			userFeatureProbabilities = new HashMap<>();
			featureCount = new HashMap<>();

		}
		void train(){
			
			for(int item:ratedItems){
				double rating = itemsRatings.get(item); // rating for a given item
				boolean userLiked = rating >= ratingThreshold ? true : false;	
				List<Integer> itemFeatures = m_featureMatrix.row(item).getIndexList(); // all features of a given item
				
				for(int feature:itemFeatures){
					if (!featureCount.containsKey(feature)){ //If we've never seen the feature before then we set the counts to 0s 
						double[] counts = {0,0};
						featureCount.put(feature, counts);
					}
					//Otherwise, we increase the count of the given feature in the corresponding column i.e. LIKE or NOT_LIKE	
					if(userLiked){
						featureCount.get(feature)[LIKE]++;

					}
					else{
						featureCount.get(feature)[NOT_LIKE]++;
					}
				}
				

			}
			//For all features observed by the user...
			for(int feature:featureCount.keySet()){
				double[] featureProbs = new double[2];
				double currentFeatureCountLike = featureCount.get(feature)[LIKE];
				double currentFeatureCountNotLike = featureCount.get(feature)[NOT_LIKE];
				
				
				double featureProbLike = (currentFeatureCountLike + alpha) / (numLiked + (2 * alpha));
				double featureProbNotLike = (currentFeatureCountNotLike + alpha) / (numNotLiked + (2 * alpha));

				featureProbs[LIKE] = featureProbLike;
				featureProbs[NOT_LIKE] = featureProbNotLike;

				userFeatureProbabilities.put(feature, featureProbs); //Add that feature along with its probabilities to the HashMap of probabilities 
					
			}
			//Setting up LaPlace variable for a given user 
			double unseenFeatureProbLike = alpha / (numLiked + (2 * alpha));
			double unseenFeatureProbNotLike = alpha / (numNotLiked + (2 * alpha));
			double[] unseenProbs = {unseenFeatureProbLike,unseenFeatureProbNotLike};
			userFeatureProbabilities.put(NOT_SEEN, unseenProbs);
		}
		public String toString(){
			StringBuilder b = new StringBuilder();
			b.append("Number of Items Liked: " + numLiked + '\n');
			b.append("Number of Items Not Liked: " + numNotLiked + '\n');
			b.append("P(L): " + pLike + '\n');
			b.append("P(~L): " + pNotLike + '\n');
			b.append("Feature count: " + dictToString(featureCount) + '\n');
			b.append("Feature probabilities: " + dictToString(userFeatureProbabilities));
			return b.toString();
			
		}
		String dictToString(HashMap<Integer,double[]> dict){
			StringBuilder b = new StringBuilder();
			b.append("{");
			for(Map.Entry<Integer, double[]> e:dict.entrySet()){
				b.append(" Feature " + e.getKey() + " : ["+e.getValue()[0] + ", " + e.getValue()[1] + "],");
			}
			b.append("}");
			return b.toString();
		}
		
	}

}

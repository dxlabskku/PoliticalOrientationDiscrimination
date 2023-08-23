# DetectingMediaSource

In this paper, to gain a more detailed understanding of whether it is possible to distinguish biases through news articles or differentiate media outlets themselves, we collected an extensive amount of news articles spanning over five years. To efficiently analyze this vast amount of data, we applied machine learning techniques.

RQ1: Can the political orientation of a news agency be determined solely based on the content of an article?                       
RQ2: Can the media outlets be determined solely based on the content of an article?


# Data

For data collection, we crawled entire articles posted in the subcategories of diplomacy/defense within the politics section of the following news sources. To ensure a balanced collection across conservative and progressive administrations, the data collection period was set from January 1, 2017, to April 25, 2023. The total collected dataset amounted to 454,255 articles. After undergoing various preprocessing steps, we utilized a final set of 36,660 conservative and 14,348 progressive articles. Specifically, this included 18,575 articles from Chosun Ilbo, 13,364 from JoongAng Ilbo, 4,721 from Dong-A Ilbo, 7,361 from Hankyoreh, and 6,987 from Kyunghyang Shinmun for our analysis.

# Result
- **Test accuracy for all feature and models** 

   The test results are average of five experiment runs. Most of the results report above 99% accuracy.
 
   ![image](https://user-images.githubusercontent.com/117256746/220046859-029d5d67-cc4e-4428-a070-377882d1dab7.png)

- **Test time per case**

   The table below shows the test time each time segment for each feature extraction methods for all models. As you can see, every case took less than 0.03 seconds, showing that they can be used for real-time detection.
   
   
   ![image](https://user-images.githubusercontent.com/117256746/220589754-b01997f6-740f-4b8b-8e0a-9f83c5ec2628.png)


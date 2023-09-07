# DetectingMediaSource

In this paper, to gain a more detailed understanding of whether it is possible to distinguish biases through news articles or differentiate media outlets themselves, we collected an extensive amount of news articles spanning over five years. To efficiently analyze this vast amount of data, we applied machine learning techniques.

RQ1: Can we identify a news outlet, based on its defense-related news articles?                       
RQ2: Can we investigate whether a specific news article is created by conservative or progressive news outlets?


# Data

For data collection, we crawled entire articles posted in the subcategories of diplomacy/defense within the politics section of the following news sources. To ensure a balanced collection across conservative and progressive administrations, the data collection period was set from January 1, 2017, to April 25, 2023. The total collected dataset amounted to 454,255 articles. After undergoing various preprocessing steps, we utilized a final set of 36,660 conservative and 14,348 progressive articles. Specifically, this included 18,575 articles from Chosun Ilbo, 13,364 from JoongAng Ilbo, 4,721 from Dong-A Ilbo, 7,361 from Hankyoreh, and 6,987 from Kyunghyang Shinmun for our analysis.

# Result
- **Detecting Political Orientation(binary-classification)** 
 
   ![image](https://github.com/dxlabskku/DetectingMediaSource/assets/126649723/e98e2a1b-4022-4cee-aea5-460f4df4c0ce)

- **Detecting Media Source(multi-classification)**
     
   ![image](https://github.com/dxlabskku/DetectingMediaSource/assets/126649723/99afb33f-7b68-4ab2-a5b2-902edf9cbcaa)



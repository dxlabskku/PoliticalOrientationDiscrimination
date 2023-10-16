# PoliticalOrientationDiscrimination

In this paper, to gain a more detailed understanding of whether it is possible to distinguish biases through news articles or differentiate media outlets themselves, we collected an extensive amount of news articles spanning over five years. To efficiently analyze this vast amount of data, we applied machine learning techniques.

RQ1: Can we identify a news outlet, based on its defense-related news articles?                       
RQ2: Can we investigate whether a specific news article is created by conservative or progressive news outlets?


# Data

For data collection, we crawled entire articles posted in the subcategories of diplomacy/defense within the politics section of the following news sources. To ensure a balanced collection across conservative and progressive administrations, the data collection period was set from January 1, 2017, to April 25, 2023. The total collected dataset amounted to 454,255 articles. After undergoing various preprocessing steps, we utilized a final set of 36,660 conservative and 14,348 progressive articles. Specifically, this included 18,575 articles from The Chosun Ilbo, 13,364 from The JoongAng Ilbo, 4,721 from The Dong-A Ilbo, 7,361 from The Hankyoreh, and 6,987 from The Kyunghyang Shinmun for our analysis.

The sample dataset is available in "news crawling data(sample).xlsx" file. We are only allowed to distribute the data for the research purpose, if you want to achieve full datasets, please complete the request form @ https://docs.google.com/forms/d/e/1FAIpQLSc72tNn4QPWYXRcUOv6D8E6zzR0a4IHCS_Qo0Z9NYfkrxw6dw/viewform?usp=sf_link

# Result
- **Result of News Outlets Classification** 
 
   ![multyresult](https://github.com/dxlabskku/PoliticalOrientationDiscrimination/assets/126649723/46673e87-5f28-4cf4-8aa6-d1566f69c3cc)

- **Result of Political Orientation Classification**
     
   ![binaryresult](https://github.com/dxlabskku/PoliticalOrientationDiscrimination/assets/126649723/1a9f60a2-0127-4a74-be9a-f754b7723d03)

## Reference
```
@article{???,
  title={Data-driven Approaches into Political Orientation and News Outlet Discrimination: the Case of News Articles in South Korea},
  author={Lee, Jungkyun and Cha, Junyeop and Park, Eunil},
  journal={Telematics and Informatics},
  volume={???},
  number={???},
  pages={???},
  year={2023},
  publisher={Elsevier}
}
```

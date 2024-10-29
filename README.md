<h1 align="center"> Improving Tourism Accessibility with Data Analytics </h1>

## Project Overview

We're developing a travel accessibility profiling system that aims to enhance tourism services for people with disabilities. By analyzing social media data and online reviews, we're identifying key accessibility attributes and sentiments across different disability groups. Our goal is to create accessibility-aware location recommendations using advanced data science techniques, including natural language processing and sentiment analysis. This project not only contributes to more inclusive tourism but also provides valuable hands-on experience in cutting-edge data analytics and research methodologies

## Contents
- [Project Overview](#project-overview)
- [Contents](#contents)
- [Data Collection](#data-collection)
- [Data Processing](#data-processing)
- [Quadruple Extraction](#quadruple-extraction)
  - [Sentires for Quadruple Extraction](#sentires-for-quadruple-extraction)
  - [Large Language Models for Quadruple Extraction](#large-language-models-for-quadruple-extraction)
- [Recommendation Systems for Users with Disabilities](#recommendation-systems-for-users-with-disabilities)
- [Interpretability of the Recommendation System](#interpretability-of-the-recommendation-system)
- [Evaluation of the Recommendation System](#evaluation-of-the-recommendation-system)

## Data Collection
The data collection process is implemented through a specialized web crawler located in the `./web_crawler/` directory. This crawler is designed to extract hotel reviews from TripAdvisor, focusing on accessibility-related information across Australia.

To run the crawler:

```
cd ./web_crawler/
python scraper.py
```

The crawler architecture is illustrated in `./img/crawler.png`.

Our dataset encompasses:
- 880,571 reviews
- 627,810 unique users
- 2,013 hotels

This extensive dataset provides valuable insights into the accessibility perceptions of various disability groups regarding tourism services.

## Data Processing
The data processing module, located in `./data_filtering/`, is responsible for extracting disability-related reviews from a large corpus of general business reviews. This critical step ensures the dataset's relevance to accessibility research.

Key features:

1. Keyword-based Filtering: Implements a robust algorithm to identify reviews containing disability-related terms and phrases.

2. One-class SVM Model: Utilizes machine learning to detect subtle patterns indicative of disability-related content, even in the absence of explicit keywords.

This module effectively addresses the challenges of  identifying nuanced disability-related content, providing a solid foundation for subsequent analysis.

## Quadruple Extraction
### Sentires for Quadruple Extraction
### Large Language Models for Quadruple Extraction


## Recommendation Systems for Users with Disabilities

## Interpretability of the Recommendation System

## Evaluation of the Recommendation System

Recruitment Assisting Buddy Engine.


Problem Statement.

•	Recruitment of technical talents is challenging for recruiters and HR professionals as they often do not know what specific technical terms mean. 

•	Therefore, they struggle with assessing which specific skills are generally required for specific job descriptions. 


•	The usual process for recruiters is to search through CVs looking for specific skills which increases the difficulty.

Approach

•	Crawled the data from all the three websites for python, ml and java considering them as Primary CSV.

•	Crawled the data for nontechnical sectors such as Banking, Finance from all the three websites considering them as secondary Csv.

•	Cleaned data from both primary and secondary csv by removing stop words and punctuations.

•	Then compared the complete 'text' of primary csv with the complete text of secondary csv. So, after comparison, we removed the words which were common in both the csv. (Non-IT terms such as salary, team).


Data Extraction

•	Extracting the data from below mentioned websites using python libraries such as Beautiful Soup, Selenium, Scrapy etc.

•	The combination of Selenium and Beautiful Soup will do the job of dynamic scraping.

•	Selenium automates web browser interaction from python. Hence the data fetched by JavaScript links can be made available by automating the button clicks with Selenium and then can be extracted by Beautiful Soup.

•	Selenium is also used to navigate to next page in the website.


Data Cleaning

•	Cleaning and pre-processing of the extracted data from the mentioned websites such as karriere.at, metajob.at, itstellen.at.

•	Removed all the stop words, punctuations and lower cased all the text.

•	Lemmatization is responsible for grouping different forms of words into the root form, having the same meaning.


Model Training.

count vectorizer.


•	Removed the redundant words after merging and comparing the primary and secondary csv.

•	Removal of common words after intersection.

•	Implementation of count vectorizer, which converts text documents to vector of token counts.

•	This approach helped us to get the top 15 skills required for the job title.


Steps to follow:

•	Run main.py file in Command Prompt

•	Add the Desired Job Title/Profile.

•	The code will then run automatically for the entered job profile in command prompt as:

1.	Data Extraction from all the three websites. (Karriere.at,IT_Stellen,MetaJobs)We have Stored them in CSV format for reference if needed later, but we will be calling it as Df so that it can run automatically for next steps.

2.	After Data collection it starts to explore the CSV.

3.	Then data cleaning and model training process using count vectorizer will begin.

•	The top 15 skills required for the job title will be displayed as output.


  

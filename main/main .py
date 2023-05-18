#!/usr/bin/env python
# coding: utf-8

# # Karriere

# In[1]:


from selenium import webdriver
from selenium.webdriver.common.by import By

options = webdriver.ChromeOptions()
options.add_argument("-headless")
options.add_argument("-no-sandbox")
options.add_argument("-disable-dev-shm-usage")

driver = webdriver.Chrome("chromedriver", options=options)




# job = "Machine Learning"
job = input("Name of the job \n")
url ="https://www.karriere.at/jobs?keywords=" + job + "&focusResults=true"

driver.get(url)


# In[2]:


print("EXTRACTING DATA FROM KARRIERE")


# In[3]:


from selenium.webdriver.common.keys import Keys
import time

page = 1
while True:
  try:
    continueBtn = driver.find_element(By.CLASS_NAME, "m-loadMoreJobsButton__button")
  except Exception:  
    break

  continueBtn.send_keys(Keys.ENTER)
  page += 1
  print("Page: " + str(page))
  time.sleep(1)


# In[4]:


links = driver.find_elements(By.CLASS_NAME, "m-jobsListItem__titleLink")
print("Count: " + str(len(links)))
urls = []

for link in links:
  urls.append(link.get_attribute("href"))


# In[5]:


from bs4 import BeautifulSoup
import re

def clean(html):
  soup = BeautifulSoup(html, "lxml")
  body = soup.find("body").text.strip()
  return re.sub(r'[\ \n]{2,}', ' ', body)


# In[6]:


data = []
count = 1

while count < len(urls):
  url = urls[count]
  print("Link: " + str(count))

  driver.get(url)
  title = driver.title

  try:
    iFrame = driver.find_element(By.CLASS_NAME, "m-jobContent__iFrame")
  except Exception:  
    print('Pause...')
    time.sleep(30)
    continue
    
  html = iFrame.get_attribute('srcdoc')
  data.append({
      'title': title,
      'text': clean(html),
      'url': url
  })

  count += 1


# In[7]:


import pandas as pd

df1 = pd.DataFrame(data)
df1.to_csv("karriere1.csv", encoding="utf-8") 


# In[8]:


df1


# # IT Stellen

# In[9]:


from selenium import webdriver
from selenium.webdriver.common.by import By

options = webdriver.ChromeOptions()
options.add_argument("-headless")
options.add_argument("-no-sandbox")
options.add_argument("-disable-dev-shm-usage")

driver = webdriver.Chrome("chromedriver", options=options)

# job = "Machine Learning"
url ="https://www.itstellen.at/jobs?keywords=" + job + "&locations=&page="

driver.get(url)


# In[10]:


print("EXTRACTING DATA FROM IT_STELLEN")


# In[11]:


from selenium.webdriver.common.keys import Keys
import time

urls = []
page = 1
tries = 0


# In[12]:


while True:
  driver.get(url + str(page))
  links = driver.find_elements(By.CSS_SELECTOR, ".jobInformation h3 a")
    
  if len(links) < 1: 
    if tries > 2:
      break
    else:
      time.sleep(20)
      tries += 1
      continue

  print("Page " + str(page) + ": " + str(len(links)))    
  for link in links:
      urls.append(link.get_attribute("href"))

  tries = 0
  page += 1
  time.sleep(10)


# In[13]:


from bs4 import BeautifulSoup
import re

def clean(html):
  soup = BeautifulSoup(html, "lxml")
  body = soup.find("body").text.strip()
  return re.sub(r'[\ \n]{2,}', ' ', body)


# In[14]:


data = []
count = 1

while count < len(urls):
  url = urls[count]
  print("Link: " + str(count))

  driver.get(url)
  title = driver.title.replace(" – itstellen.at", "")

  try:
    content = driver.find_element(By.CLASS_NAME, "jobContent").get_attribute('innerHTML')
  except Exception:  
    print('Pause...')
    time.sleep(30)
    continue
    
  data.append({
      'title': title,
      'text': clean(content),
      'url': url
  })

  count += 1
  time.sleep(5)


# In[15]:


import pandas as pd

df2 = pd.DataFrame(data)
df2.to_csv('IT-Stellen1.csv')


# In[16]:


df2


# # Meta Jobs

# In[17]:


print("EXTRACTING DATA FROM META JOBS")


# In[18]:


from selenium import webdriver
import time
from selenium.webdriver.common.by import By
import pandas as pd
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore")


# In[19]:


def click_next_page(driver):
    """
    This function is responsible for clicking the 'next' button to navigate to the next page.
    We are using xpath here since static id is missing.
    For locating the exact element, below logic is used:

    1. extract all buttons in the footer via xpath.
    2. extract the last element from the list of buttons.
    3. this is done since the array length changes when we navigate to page 4 onwards.
    4. 'next' button would always be the last in the list so we fetch it via -1 index.
    """

    buttons = driver.find_elements(By.XPATH,
                                   '/html/body/div/div[2]/div[2]/div/div/div[1]/div[2]/div/div/div[2]/div[1]/div[2]/div/button')
    next_button = buttons[-1]
    next_disabled = next_button.get_property("disabled")

    print('Next page button disabled? {disabled}'.format(disabled=next_disabled))
    if next_button.is_enabled() or not next_disabled:
        next_button.click()
        time.sleep(2)
        return True
    else:
        return False


# In[20]:


def click_page_size(driver, count):
    """
    This function is responsible for changing the default number of entries on the page.
    By default we have 10 entries, that could lead to alot of pages to be scraped.
    We change this to 50 (by clicking on it twice).
    This reduces the number of pages to be extracted.
    """

    for i in range(count):
        driver.find_element(By.XPATH, '/html/body/div/div[2]/div[2]/div/div/div[1]/div[1]/button[3]').click()
        time.sleep(1)


# In[21]:


def extract_metadata_for_job(job):
    """
    This function is responsible for extracting the metadata for each job entry.
    metadata consists of title, description and job URL.

    :param job: DOM element for the job entry.
    :return: dict containing metadata info.
    """
    metadata = None
    job = job.find("div", class_="job-text-div")
    try:
        title = job.find("a", class_="resultUrl hyphenate").text
        desc = job.find("div", class_="snippet hyphenate").text
        job_url = base_url + job.find("a", class_="resultUrl hyphenate").get("href")
        metadata = {"title": title, "text": desc, "url": job_url}
    except:
        pass  # silently ignore any parsing failure
    finally:
        return metadata


# In[24]:


def extract_metadata_for_page(driver):
    """
    This function is responsible for extracting the metadata for all the jobs listed on this page.
    :param driver:
    :return: dataframe with metadata for all job entries found on the page.
    """

    df = pd.DataFrame(columns=['title', 'text', 'url'])
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    jobs = soup.find("div", class_="joblist").findChildren()

    for job in jobs:
        metadata = extract_metadata_for_job(job)
        if metadata:
            df = df.append(metadata, ignore_index=True)
    return df
if __name__ == '__main__':

    base_url = 'https://www.metajob.at' # This is needed to concat with job URL
    it_url = 'https://www.metajob.at/' + job

    try:
        next_page_available = True
        counter = 1
        df = pd.DataFrame(columns=['title', 'text', 'url'])

        driver = driver = webdriver.Chrome(executable_path='https://www.metajob.at')
        driver.maximize_window()
        driver.get(it_url)
        time.sleep(3)  # TO-DO remove all hard-coded sleep with selenium way of waiting.
        click_page_size(driver, 2)

        while next_page_available:
            print('Starting extraction for page {page}'.format(page=counter))
            df = df.append(extract_metadata_for_page(driver))
            next_page_available = click_next_page(driver)
            counter += 1

        print('Total entries extracted: {entries}'.format(entries=df.shape[0]))
        df.to_csv('javautomate.csv', index=False)

    except:
        print('Issue in opening web page')
    finally:
        driver.quit()


# In[25]:


df


# In[26]:


print("DATA FROM  KARRIERE, IT_STELLEN AND META JOBS IS COLLECTED")


# In[27]:


Df = pd.concat([df1, df2,df])


# In[28]:


Df


# In[29]:


Df.to_csv("primary.csv", encoding="utf-8") 


# # DATA CLEANING

# In[30]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# In[31]:


import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[32]:


import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')


# In[33]:


x = Df.shape
x 


# In[34]:


def cleanDescription(descText):
    descText = descText.lower() #converting to lowercase
    descText = re.sub('http\S+\s*', ' ', descText)  # remove URLs
    descText = re.sub('RT|cc', ' ', descText)  # remove RT and cc
    descText = re.sub('#\S+', '', descText)  # remove hashtags
    descText = re.sub('@\S+', '  ', descText)  # remove mentions
    descText = re.sub(r'[0-9]', '', descText) # remove numbers
    descText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', descText)  # remove punctuations
    #descText = re.sub(r'[^\x00-\x7f]',r' ', descText) # remove non Non ASCII CHarecter
    descText = re.sub('\s+', ' ', descText)  # remove extra whitespace
    return descText 


# In[35]:


Df['cleaned_re'] = Df['text'].apply(lambda x:cleanDescription(x))
Df.head() 


# In[36]:


def text_clean_2(text):
  stop_words = set(stopwords.words('english'))
  word_tokens = word_tokenize(text)
  filtered_sentence = []
  wnl = WordNetLemmatizer()
 
  for w in word_tokens:
      if w not in stop_words:
          filtered_sentence.append(w)
  lemmatized_string = ' '.join([wnl.lemmatize(words) for words in filtered_sentence])
  return lemmatized_string         
  #return filtered_sentence


# In[37]:


Df['cleaned_text'] = Df['cleaned_re'].apply(lambda x:text_clean_2(x))
Df.head(2)


# In[38]:


row_no = len(Df)
print(row_no)


# In[39]:


index = pd.Index(range(0,row_no,1))
Df = Df.set_index(index)
Df


# In[40]:


corpus = " "
for i in range(0, row_no):
    corpus = corpus + Df["cleaned_text"][i]


# In[41]:


import nltk
from nltk.corpus import stopwords
sw_nltk = stopwords.words('german')

alpha_word = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n' 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
#new_word = ['un', 'ber', 'sowie','erfahrung','al','innen','de','au','sungen','bereich', 'bieten', 'kenntnisse','gute','benefit', 'ab','service', 'arbeiten','kunden','flexible', 'unternehmen','bewerben', 'wien', 'office','experience','eur', 'technologien','unterst','unserer', 'ausbildung', 'position', 'qualifikation', 'work', 'job', 'mitarbeiter', 'aufgaben','bi', 'teil','design','online', 'freuen','glichkeiten','web', 'vollzeit', 'weiterentwicklung', 'arbeitszeiten','u', 'h', 'gmbh', 'working', 'bewerbung','test', 'pers', 'brutto','suchen','brz','hast','qualit','glich','htl','je','home', 'business','mehr','system','glichkeit', 'technology','engineering', 'solution', 'gerne','gehalt','vorteil','gro','technische','fh','neue', 'gemeinsam','skill','umsetzung','abgeschlossene','homeoffice','sterreich','berzahlung','www','international','dabei', 'ngig','bzw', 'nnen', 'partner','know', 'zusammenarbeit','anforderungen','tzen','gesch','etc', 'higkeit', 'recruiting', 'zukunft', 'neuen','englischkenntnisse', 'interesse','tzung', 'level', 'hrung','linz','knowledge','company', 'time','mitarbeit','integration','abh', 'full', 'weitere', 'profil','new','event', 'applikationen','erste','erfahrungen','austria','internationalen','verg','bringen','standort','kannst', 'basis','sofort','menschen','leben', 'verst','life','group', 'gr','weiterbildung', 'stack','verf','tzt', 'net', 'salary','uni','monat','erm', 'bereichen','graz','mindestens','kontakt','arbeit','environment','anwendungen']
new_word = ['berufserfahrung','un', 'ber', 'sowie','erfahrung','al','innen','de','au','sungen','bereich', 'bieten', 'kenntnisse','gute','benefit', 'ab','service', 'arbeiten','kunden','flexible', 'unternehmen','bewerben', 'wien', 'office','experience','eur', 'technologien','unterst','unserer', 'ausbildung', 'position', 'qualifikation', 'work', 'job', 'mitarbeiter', 'aufgaben','bi', 'teil','design','online', 'freuen','glichkeiten','web', 'vollzeit', 'weiterentwicklung', 'arbeitszeiten','u', 'h', 'gmbh', 'working', 'bewerbung','test', 'pers', 'brutto','suchen','brz','hast','qualit','glich','htl','je','home', 'business','mehr','system','glichkeit', 'technology','engineering', 'solution', 'gerne','gehalt','vorteil','gro','technische','fh','neue', 'gemeinsam','skill','umsetzung','abgeschlossene','homeoffice','sterreich','berzahlung','www','international','dabei', 'ngig','bzw', 'nnen', 'partner','know', 'zusammenarbeit','anforderungen','tzen','gesch','etc', 'higkeit', 'recruiting', 'zukunft', 'neuen','englischkenntnisse', 'interesse','tzung', 'level', 'hrung','linz','knowledge','company', 'time','mitarbeit','integration','abh', 'full', 'weitere', 'profil','new','event', 'applikationen','erste','erfahrungen','austria','internationalen','verg','bringen','standort','kannst', 'basis','sofort','menschen','leben', 'verst','life','group', 'gr','weiterbildung', 'stack','verf','tzt', 'net', 'salary','uni','monat','erm', 'bereichen','graz','mindestens','kontakt','arbeit','environment','anwendungen','liegt','st', 'erstellung','part','pr','mehrj','spannende', 'mindestgehalt','art','rund','ten','idealerweise','softwarel','bereitschaft','arbeitest','stellen','com','hrige','datenbanken','weltweit','ro','arbeitsweise','end', 'gesundheit','requirement','studium','qualifikationen', 'gt', 'selbstst','bringst','expert', 'arbeitsumfeld','karriere','jahre', 'angebot','regelm','bereits','offer','modernen', 'konzeption','umgang', 'ideen', 'onboarding', 'ren', 'architektur', 'implementierung', 'frau', 'ndige','high', 'good', 'customer', 'tech', 'chancengleichheit','en', 'user', 'inkl','mail','verantwortung','well''informationen','neuer','leidenschaft','rolle', 'erwartet','mobile','durchf','zahlreiche','arbeitsplatz', 'jahren','beim', 'kollektivvertrag','nstigungen','year','pro','top','freude', 'per','wiener', 'salzburg','weiterbildungen','process','gut','ffentliche','opportunity', 'ndig','hohe','betrieb','wissen','anbindung','glichen','kund','looking','aktiv','lebenslauf','welt', 'diverse','themen','wort','weiterbildungsm','lage','gesamten', 'balance', 'kv', 'sport', 'gross', 'geh', 'bietet','optimierung', 'like', 'platform','vielfalt','zusammen', 'apply', 'performance', 'schrift','stehen','ort','bewirb', 'gen', 'hour', 'based', 'quality', 'ansprechpartner','fundierte','state','innovativen','people', 'ansprechperson','tze', 'schwerpunkt','beko', 'qualification','individuelle','jahresbruttogehalt']

sw_nltk.extend(alpha_word)
sw_nltk.extend(new_word)

text = corpus
words = [word for word in text.split() if word.lower() not in sw_nltk]
sentence1 = " ".join(words)

#print(new_text)
print("Old length: ", len(text))
print("New length: ", len(sentence1))


# In[42]:


import pandas as pd
df_Banking = pd.read_csv('Secondary.csv')
df_Banking.head()


# In[43]:


def cleanDescription(descText):
    descText = descText.lower() #converting to lowercase
    descText = re.sub('http\S+\s*', ' ', descText)  # remove URLs
    descText = re.sub('RT|cc', ' ', descText)  # remove RT and cc
    descText = re.sub('#\S+', '', descText)  # remove hashtags
    descText = re.sub('@\S+', '  ', descText)  # remove mentions
    descText = re.sub(r'[0-9]', '', descText) # remove numbers
    descText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', descText)  # remove punctuations
    descText = re.sub(r'[^\x00-\x7f]',r' ', descText) # remove non Non ASCII CHarecter
    descText = re.sub('\s+', ' ', descText)  # remove extra whitespace
    return descText 

df_Banking['cleaned_re'] = df_Banking['text'].apply(lambda x:cleanDescription(x))
df_Banking.head()


# In[44]:


def text_clean_2(text):
  stop_words = set(stopwords.words('english'))
  word_tokens = word_tokenize(text)
  filtered_sentence = []
  wnl = WordNetLemmatizer()
 
  for w in word_tokens:
      if w not in stop_words:
          filtered_sentence.append(w)
  lemmatized_string = ' '.join([wnl.lemmatize(words) for words in filtered_sentence])
  return lemmatized_string         
  #return filtered_sentence
df_Banking['cleaned_text'] = df_Banking['cleaned_re'].apply(lambda x:text_clean_2(x))
df_Banking.head(2)


# In[45]:


row_no = len(df_Banking)
row_no


# In[46]:


#getting the entire resume text
Bank_corpus=" "
for i in range(0,row_no):
    Bank_corpus= Bank_corpus+ df_Banking["cleaned_text"][i]


# In[47]:


import nltk
from nltk.corpus import stopwords
sw_nltk = stopwords.words('german')

alpha_word = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n' 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
#new_word = ['un', 'ber', 'sowie','erfahrung','al','innen','de','au','sungen','bereich', 'bieten', 'kenntnisse','gute','benefit', 'ab','service', 'arbeiten','kunden','flexible', 'unternehmen','bewerben', 'wien', 'office','experience','eur', 'technologien','unterst','unserer', 'ausbildung', 'position', 'qualifikation', 'work', 'job', 'mitarbeiter', 'aufgaben','bi', 'teil','design','online', 'freuen','glichkeiten','web', 'vollzeit', 'weiterentwicklung', 'arbeitszeiten','u', 'h', 'gmbh', 'working', 'bewerbung','test', 'pers', 'brutto','suchen','brz','hast','qualit','glich','htl','je','home', 'business','mehr','system','glichkeit', 'technology','engineering', 'solution', 'gerne','gehalt','vorteil','gro','technische','fh','neue', 'gemeinsam','skill','umsetzung','abgeschlossene','homeoffice','sterreich','berzahlung','www','international','dabei', 'ngig','bzw', 'nnen', 'partner','know', 'zusammenarbeit','anforderungen','tzen','gesch','etc', 'higkeit', 'recruiting', 'zukunft', 'neuen','englischkenntnisse', 'interesse','tzung', 'level', 'hrung','linz','knowledge','company', 'time','mitarbeit','integration','abh', 'full', 'weitere', 'profil','new','event', 'applikationen','erste','erfahrungen','austria','internationalen','verg','bringen','standort','kannst', 'basis','sofort','menschen','leben', 'verst','life','group', 'gr','weiterbildung', 'stack','verf','tzt', 'net', 'salary','uni','monat','erm', 'bereichen','graz','mindestens','kontakt','arbeit','environment','anwendungen']
new_word = ['berufserfahrung','un', 'ber', 'sowie','erfahrung','al','innen','de','au','sungen','bereich', 'bieten', 'kenntnisse','gute','benefit', 'ab','service', 'arbeiten','kunden','flexible', 'unternehmen','bewerben', 'wien', 'office','experience','eur', 'technologien','unterst','unserer', 'ausbildung', 'position', 'qualifikation', 'work', 'job', 'mitarbeiter', 'aufgaben','bi', 'teil','design','online', 'freuen','glichkeiten','web', 'vollzeit', 'weiterentwicklung', 'arbeitszeiten','u', 'h', 'gmbh', 'working', 'bewerbung','test', 'pers', 'brutto','suchen','brz','hast','qualit','glich','htl','je','home', 'business','mehr','system','glichkeit', 'technology','engineering', 'solution', 'gerne','gehalt','vorteil','gro','technische','fh','neue', 'gemeinsam','skill','umsetzung','abgeschlossene','homeoffice','sterreich','berzahlung','www','international','dabei', 'ngig','bzw', 'nnen', 'partner','know', 'zusammenarbeit','anforderungen','tzen','gesch','etc', 'higkeit', 'recruiting', 'zukunft', 'neuen','englischkenntnisse', 'interesse','tzung', 'level', 'hrung','linz','knowledge','company', 'time','mitarbeit','integration','abh', 'full', 'weitere', 'profil','new','event', 'applikationen','erste','erfahrungen','austria','internationalen','verg','bringen','standort','kannst', 'basis','sofort','menschen','leben', 'verst','life','group', 'gr','weiterbildung', 'stack','verf','tzt', 'net', 'salary','uni','monat','erm', 'bereichen','graz','mindestens','kontakt','arbeit','environment','anwendungen','liegt','st', 'erstellung','part','pr','mehrj','spannende', 'mindestgehalt','art','rund','ten','idealerweise','softwarel','bereitschaft','arbeitest','stellen','com','hrige','datenbanken','weltweit','ro','arbeitsweise','end', 'gesundheit','requirement','studium','qualifikationen', 'gt', 'selbstst','bringst','expert', 'arbeitsumfeld','karriere','jahre', 'angebot','regelm','bereits','offer','modernen', 'konzeption','umgang', 'ideen', 'onboarding', 'ren', 'architektur', 'implementierung', 'frau', 'ndige','high', 'good', 'customer', 'tech', 'chancengleichheit','en', 'user', 'inkl','mail','verantwortung','well''informationen','neuer','leidenschaft','rolle', 'erwartet','mobile','durchf','zahlreiche','arbeitsplatz', 'jahren','beim', 'kollektivvertrag','nstigungen','year','pro','top','freude', 'per','wiener', 'salzburg','weiterbildungen','process','gut','ffentliche','opportunity', 'ndig','hohe','betrieb','wissen','anbindung','glichen','kund','looking','aktiv','lebenslauf','welt', 'diverse','themen','wort','weiterbildungsm','lage','gesamten', 'balance', 'kv', 'sport', 'gross', 'geh', 'bietet','optimierung', 'like', 'platform','vielfalt','zusammen', 'apply', 'performance', 'schrift','stehen','ort','bewirb', 'gen', 'hour', 'based', 'quality', 'ansprechpartner','fundierte','state','innovativen','people', 'ansprechperson','tze', 'schwerpunkt','beko', 'qualification','individuelle','jahresbruttogehalt']

sw_nltk.extend(alpha_word)
sw_nltk.extend(new_word)

text = Bank_corpus
words = [word for word in text.split() if word.lower() not in sw_nltk]
sentence2 = " ".join(words)

#print(new_text)
print("Old length: ", len(text))
print("New length: ", len(sentence2 ))


# In[48]:


print("Data Cleaning ends")


# # Compairing Primary and Secondary CSV

# In[49]:


print("Comparing primary and secondary csv")


# In[50]:


def common_words(sentence1, sentence2):
    # split the sentences into lists of words
    words1 = sentence1.split()
    words2 = sentence2.split()
    
    # use set intersection to get the common words
    common = set(words1).intersection(words2)
    
    # convert the set back to a list and return it
    return list(common)

# print(common_words(sentence2, sentence1))
Common_words = common_words(sentence1, sentence2)
Common_words


# In[51]:


UnwantedWords = ['abhängig', '€','deutsch','unterstützung', '–','„','englisch','möglich', 'mehrjährige', 'unterstützt', 'unterstützen','durchführung', 'österreich','möglichkeit', 'lösungen', 'überzahlung', '•','support', 'react','rest','closely','open','entwickler','architekt','microservice', 'programmierung','logistik','expertinnen','exzellente','logistik','intralogistik','entwicklern','entwerfen','softwareentwickler','objektorientierter','mehrjährige','architekturen','language','restful','…sie','gesammelt','tu','entwicklungsteam','entwicklerin','openshift','openshift','basierend','boot','namhafte','erwartungen','moderner','webbasierten','se','develop','unterstützung','kundenspezifischen','webservices','neuentwicklung','strong','öffentliche','similar','heben','…wir','ehestmöglichen','thing','produzierenden','einkaufen','leader','exzellenten','junit','suite','wer','tention','option','weiterentwicklungen','plattformen','scalable','ording','seit', 'technischen', 'produkte','pocket','mitwirkt','mentoring','reward','loyalty','ansätze','inbetriebnahme','modernster','basierten','hibernate','act','computer','funktionalen','hagenberg','revolutionieren','skalierbaren','kundenanforderungen','entwicklerteams','aktueller','•','neues','industrie','…du','critical','’','energie','evergrowing','vorhandenen','healthcare','source','unterstützt','vienna','feature','senior', 'unterstützen','durchführung', 'ci', 'möchten', 'moderne', 'essenszuschuss', 'infrastruktur', 'stelle', 'übernehmen', 'weiterbildungsmöglichkeiten', 'daher','österreich','möglichkeit', 'lösungen', 'überzahlung']

sw_nltk.extend(Common_words)
sw_nltk.extend(UnwantedWords)

text = corpus
words = [word for word in text.split() if word.lower() not in sw_nltk]
NewCleanedText = " ".join(words)

#print(new_text)
print("Old length: ", len(sentence1))
print("New length: ", len(NewCleanedText ))


# In[52]:


import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
word_cloud = WordCloud(collocations = False, background_color = 'white').generate(NewCleanedText)


# In[53]:


plt.figure(figsize = (12,12), facecolor = None)
plt.imshow(word_cloud)
plt.axis("off")
plt.tight_layout(pad = 0)


# In[54]:


import pandas as pd

# Sample sentence
sentence = NewCleanedText

# Split the sentence into words
words = sentence.split()

# Calculate the number of rows required
num_rows = (len(words) + 199) // 200

# Create a DataFrame with the required number of rows
df_Words = pd.DataFrame({'Words': ['']*num_rows})

# Populate each row of the DataFrame with up to 200 words
for i in range(num_rows):
    start_index = i * 200
    end_index = min((i+1) * 200, len(words))
    df_Words.iloc[i] = ' '.join(words[start_index:end_index])

# Print the resulting DataFrame
print(df_Words)


# In[58]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


# Create a CountVectorizer object to convert the job descriptions to a matrix of token counts
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df_Words["Words"])

# Get the feature names (i.e. the words/tokens)
feature_names = vectorizer.get_feature_names_out()

# Get the sum of token counts for each feature (i.e. the number of times each word appears in the job descriptions)
word_counts = X.sum(axis=0)

# Create a dictionary mapping feature names to their counts
word_counts_dict = dict(zip(feature_names, word_counts.flat))

# Sort the dictionary by value (i.e. word count) in descending order
sorted_word_counts = sorted(word_counts_dict.items(), key=lambda x: x[1], reverse=True)

# Extract the most common job skills
n_common_skills = 15
common_skills = [skill[0] for skill in sorted_word_counts[:n_common_skills]]
print("The required skills are")
print(common_skills)


# In[1]:


print("END OF CODE final output generated")


# In[ ]:


df_1 = pd.DataFrame(common_skills)
print(df_1)


# In[ ]:


df_1.to_csv('output.csv')


# In[ ]:


print("RESULTS ARE STORED IN OUTPUT.CSV FILE")


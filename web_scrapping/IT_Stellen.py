#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install chromium-chromedriver')
# !cp /usr/lib/chromium-browser/chromedriver /usr/bin
get_ipython().system('pip install selenium')


# In[1]:


from selenium import webdriver
from selenium.webdriver.common.by import By

options = webdriver.ChromeOptions()
options.add_argument("-headless")
options.add_argument("-no-sandbox")
options.add_argument("-disable-dev-shm-usage")

driver = webdriver.Chrome("chromedriver", options=options)

job = "Java%20Entwickler"
url ="https://www.itstellen.at/jobs?keywords=" + job + "&locations=&page="


# In[2]:


from selenium.webdriver.common.keys import Keys
import time

urls = []
page = 1
tries = 0

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


# In[3]:


from bs4 import BeautifulSoup
import re

def clean(html):
  soup = BeautifulSoup(html, "lxml")
  body = soup.find("body").text.strip()
  return re.sub(r'[\ \n]{2,}', ' ', body)


# In[4]:


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


# In[5]:


import pandas as pd

df = pd.DataFrame(data)
df.to_csv('IT-Stellen.csv')


# In[ ]:





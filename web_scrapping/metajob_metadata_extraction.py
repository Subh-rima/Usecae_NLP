from selenium import webdriver
import time
from selenium.webdriver.common.by import By
import pandas as pd
from bs4 import BeautifulSoup
import warnings

warnings.filterwarnings("ignore")


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
        metadata = {"title": title, "description": desc, "job_url": job_url}
    except:
        pass  # silently ignore any parsing failure
    finally:
        return metadata


def extract_metadata_for_page(driver):
    """
    This function is responsible for extracting the metadata for all the jobs listed on this page.
    :param driver:
    :return: dataframe with metadata for all job entries found on the page.
    """

    df = pd.DataFrame(columns=['title', 'description', 'job_url'])
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    jobs = soup.find("div", class_="joblist").findChildren()

    for job in jobs:
        metadata = extract_metadata_for_job(job)
        if metadata:
            df = df.append(metadata, ignore_index=True)
    return df


if __name__ == '__main__':

    base_url = 'https://www.metajob.at' # This is needed to concat with job URL
    java_dev_url = 'https://www.metajob.at/java%20developer'

    try:
        next_page_available = True
        counter = 1
        df = pd.DataFrame(columns=['title', 'description', 'job_url'])

        driver = webdriver.Firefox()
        driver.maximize_window()
        driver.get(java_dev_url)
        time.sleep(3)  # TO-DO remove all hard-coded sleep with selenium way of waiting.
        click_page_size(driver, 2)

        while next_page_available:
            print('Starting extraction for page {page}'.format(page=counter))
            df = df.append(extract_metadata_for_page(driver))
            next_page_available = click_next_page(driver)
            counter += 1

        print('Total entries extracted: {entries}'.format(entries=df.shape[0]))
        df.to_csv('java_developer_dump.csv', index=False)

    except:
        print('Issue in opening web page')
    finally:
        driver.quit()

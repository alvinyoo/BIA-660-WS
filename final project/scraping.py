"""
Final project: scraping
Collect 5,000 Job Ads for Data Scientists from Indeed.com.
Collect 5,000 Job Ads for Software Engineers in from Indeed.com.
Collect 5,000 Job Ads for Data Engineers from Indeed.com.
members: Yuchen Yao, Jiahao Li, Nuozhou Tang, Qingxi Liu, Shiying Li
"""

from selenium import webdriver
from bs4 import BeautifulSoup
import requests
import time
import csv


def scrape(link, city, tittle):
    # by BeautifulSoup
    fw = open(f"{city}_{tittle}.csv", 'a', encoding='utf8')  # output file
    writer = csv.writer(fw, lineterminator='\n')  # create a csv writer for this file

    for p in range(0, 100):
        if p != 0:
            url = link + f"&start={p*10}"
        elif p == 0:
            url = link
        # print(url)

        response = None
        for i in range(5):
            response = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_5) AppleWebKit/537.78.2 (KHTML, like Gecko)'
                              ' Version/6.1.6 Safari/537.78.2'})
            if response:
                break
            else:
                print('failed attempt', i+1)

        if not response:
            return None  # all five attempts fail

        # get text from response
        html = response.text  # read in the text from the file
        soup = BeautifulSoup(html, 'html.parser')
        # print(soup)

        jobs = soup.findAll('div', {'class': 'jobsearch-SerpJobCard unifiedRow row result'})  # get all the title divs

        for job in jobs:
            # title = job.find('h2', {'class': 'title'}).text.strip().lower()
            if tittle == 'data+scientist':
                lable = 1
            elif tittle == 'software+engineer':
                lable = 2
            else:
                lable = 3

            lis = job.findAll('div', {'class': 'summary'})
            summary = ' '.join([li.text.replace('\n', ' ').lower() for li in lis])

            writer.writerow([summary, lable])  # write to file

        time.sleep(1)

        if p % 100 == 0:
            time.sleep(120)

        nextPG = soup.findAll('span', {'class': 'np'})
        if p != 0:
            if len(nextPG) == 1:
                break

    fw.close()


def getJob(link, city, tittle):
    # by WebDriver
    fw = open(f"{city}_{tittle}.csv", 'a', encoding='utf8')  # output file
    writer = csv.writer(fw, lineterminator='\n')  # create a csv writer for this file
    driver = webdriver.Chrome("./chromedriver")

    links = []
    for p in range(0, 100):
        print(f"\tpage num: {p}")
        if p != 0:
            url = link + f"&start={p * 10}"
        elif p == 0:
            url = link

        driver.get(url)
        driver.implicitly_wait(5)

        job_card = driver.find_elements_by_xpath('//div[contains(@class,"clickcard")]')

        for job in job_card:
            links.append(job.find_element_by_xpath('.//h2[@class="title"]//a').get_attribute(name="href"))
        time.sleep(1)

        if p % 200 == 0:
            time.sleep(60)

        try:
            nextPG = driver.find_elements_by_class_name('np')
        except:
            nextPG = None
        if p != 0:
            if len(nextPG) == 1:
                break

    time.sleep(60)
    for i in range(len(links)):
        driver.get(links[i])
        driver.implicitly_wait(4)

        time.sleep(2)

        description = driver.find_element_by_xpath('//div[@id="jobDescriptionText"]').text

        if i % 20 == 0:
            time.sleep(7 * 60)

        if tittle == 'data+scientist':
            lable = 1
        elif tittle == 'software+engineer':
            lable = 2
        else:
            lable = 3
        writer.writerow([description.strip().replace('\n', '. '), lable])  # write to file

    fw.close()


def runSoup(cities, titles):
    start = time.time()

    for job in titles:
        for city in cities:
            st = time.time()
            url = f'https://www.indeed.com/jobs?q={job}&l={city}&sort=date'
            scrape(url, city, job)
            print(f"{city}-{job} finished! Run time: {time.time() - st}")
            time.sleep(5 * 60)

    print(f"run time: {time.time() - start}")


def runDriver(cities, titles):
    start = time.time()

    for job in titles:
        for city in cities:
            st = time.time()
            url = f'https://www.indeed.com/jobs?q={job}&l={city}&sort=date'
            getJob(url, city, job)
            print(f"{city}-{job} finished! Run time: {time.time() - st}")
            time.sleep(5 * 60)

    print(f"run time: {time.time() - start}")


if __name__ == '__main__':
    # cities = ['New+York', 'Seattle', 'Palo+Alto']
    # titles = ['data+scientist', 'software+engineer', 'data+engineer']
    cities = ['Los+Angeles']
    titles = ['software+engineer']

    # runSoup(cities=cities, titles=titles)
    runDriver(cities=cities, titles=titles)

# !pip3 install BeautifulSoup4

import urllib
import re
from bs4 import BeautifulSoup
import time


# https://www.wikiart.org/en/claude-monet/all-works/text-list
# https://www.wikiart.org/en/claude-monet/camille-with-a-small-dog

file_path = "/work/data/wikiart"
base_url = 'https://www.wikiart.org'
artist = '/en/claude-monet'

# get the artist's main page
url = base_url + artist + '/all-works/text-list'
artist_work_soup = BeautifulSoup(urllib.request.urlopen(url), "html.parser")

# get the main section
artist_main = artist_work_soup.find("main")
image_count = 0
artist_name = artist.split("/")[2]

# get the list of paintings and download each of them
lis = artist_main.find_all("li")
for li in lis:
    link = li.find("a")

    if link != None:
        painting = link.attrs["href"]
        # get the painting
        url = base_url + painting
        print(url)
        try:
            painting_soup = BeautifulSoup(
                urllib.request.urlopen(url), "html.parser")
        except:
            print("error retreiving page")
            continue

        # get the url
        og_image = painting_soup.find("meta", {"property": "og:image"})
        image_url = og_image["content"].split(
            "!")[0]  # ignore the !Large.jpg at the end
        print(image_url)

        save_path = file_path + "/" + artist_name + \
            "_" + str(image_count) + ".jpg"
        # download the file
        try:
            print("downloading to " + save_path)
            time.sleep(0.2)  # to avoid a 403
            urllib.request.urlretrieve(image_url, save_path)
            image_count = image_count + 1
        except Exception as e:
            print("failed downloading " + image_url, e)

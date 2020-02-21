import requests
from bs4 import BeautifulSoup
import urllib.request
import re
import shutil



home_sheets = [
    'https://www.spriters-resource.com/game_boy_advance/pokemonemerald/',
    'https://www.spriters-resource.com/game_boy_advance/pokemonfireredleafgreen/',
    'https://www.spriters-resource.com/game_boy_advance/pokemonmysterydungeonredrescueteam/',
    'https://www.spriters-resource.com/game_boy_advance/pokemonrubysapphire/'
    ]


filename_counter = 0

# Each home page corresponds to a GBA's pokemon game
for home_url in home_sheets:
    request = requests.get(home_url)
    html = request.text
    soup = BeautifulSoup(html, 'html.parser')

    # Each home pages lists tons of spritesheets
    for item in soup.find_all(
            lambda tag: tag.has_attr('href')
            and 'game_boy_advance/' in tag['href']
            and '/sheet/' in tag['href']):
        href = item['href']
        id = href.strip('/').split('/')[-1]

        # This urls gets the file, fileinfo is on headers
        download_url = 'https://www.spriters-resource.com/download/{}/'.format(id)

        # Above URL is obviously redirected to some file, so resolve the redirection
        req = urllib.request.urlopen(download_url)

        content_disposition = req.headers['content-disposition']
        server_name = re.findall("filename=(.+)", content_disposition)[0].strip('";')

        final_name = '{:04} - {}'.format(filename_counter, server_name)
        filename_counter += 1

        filepath = './data/spriters-resource/{}'.format(final_name)

        with open(filepath, 'wb') as datafile:
            print(filepath.strip('./'))
            server_data = requests.get(download_url, stream=True)
            shutil.copyfileobj(server_data.raw, datafile)

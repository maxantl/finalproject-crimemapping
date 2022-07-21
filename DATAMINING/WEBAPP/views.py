from django.shortcuts import render
import folium
from folium import plugins
import plotly.express as px
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import ngrams
import json
import requests
import io
import re
import warnings
warnings.filterwarnings('ignore')

def test(request):
    data = data_preparation()
    map1 = spatial_analysis_heat(data)
    map2 = spatial_analysis_chloro(data)
    context = {
        'map1': map1,
        'map2': map2,
    }
    return render(request,'WEBAPP/test.html',context)

def data_preparation():
    url = "https://raw.githubusercontent.com/lorensdima/datasetstest/main/type_data.csv"
    download = requests.get(url).content
    df = pd.read_csv(io.StringIO(download.decode('utf-8')))

    url = "https://raw.githubusercontent.com/lorensdima/datasetstest/main/tagalog_stopwords.csv"
    download = requests.get(url).content
    tagalog_stops = pd.read_csv(io.StringIO(download.decode('utf-8'))).a.to_list()
    stop_words = stopwords.words('english') + tagalog_stops

    def clean_tweets(text):
        text = text.lower()
        text = re.sub(r'RT @[\w]*', '', str(text))
        text = re.sub(r'@[\w]*', '', str(text))
        text = re.sub(r'#([a-zA-Z0-9_]{1,50})', '', str(text))
        text = re.sub(r'https?://[A-Za-z0-9./]*', '', str(text))
        text = re.sub(r'\n', '', str(text))
        text = re.sub(r'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]', u'', str(text))
        text = re.sub(r'[^\x00-\x7f]', '', str(text))
        text = re.sub(r'[^\w\s]', '', str(text))
        text_tokens = word_tokenize(text)
        filtered_text = [w for w in text_tokens if not w in stop_words]
        return " ".join(filtered_text)

    df['tweet'] = df['tweet'].apply(lambda x: clean_tweets(x))

    # Get geojson data for cities in the Philippine
    url = "https://raw.githubusercontent.com/lorensdima/datasetstest/main/cities.geojson"
    f = requests.get(url)
    json_city = f.json()

    id_map = {}
    city_map = {}
    province_map = {}
    region_map = {}
    index = 0
    for feature in json_city["features"]:
        # feature["id"] = feature["properties"]["ID_2"]
        feature["id"] = index
        if feature["geometry"] != None:
            id_map[feature["properties"]["NAME_1"].lower() + " " + feature["properties"]["NAME_2"].lower()] = feature[
                "id"]
            words = feature["properties"]["NAME_2"].lower().split()
            for w in words:
                city_map[w] = feature["id"]
            words = feature["properties"]["NAME_1"].lower().split()
            for w in words:
                province_map[w] = feature["id"]
            region_map[feature["properties"]["REGION"].lower().split("(")[0].strip()] = feature["id"]
        index += 1

    df['coords'] = np.NaN
    no_coords = df[['tweet', 'coords', 'type']]

    def locate_tweet(text):
        found = False
        n_gram = 1
        while not found:
            # spliting the sentence
            words = ngrams(text.split(), n_gram)
            result = ""
            text_tokens = word_tokenize(text)
            city = ""
            province = ""
            init_loc = ""
            for w in words:
                w = re.sub(r'[^a-zA-Z]', '', str(w))
                if city_map.__contains__(w) and city == "":
                    city = w
                    init_loc = w + " " + init_loc
                if province_map.__contains__(w) and province == "":
                    province = w
                    init_loc = init_loc + w
                if city != "" and province != "":
                    init_loc.strip()
                    break;
            index_loc = -1
            if id_map.__contains__(init_loc):
                index_loc = id_map[init_loc]
            elif province != "":
                index_loc = province_map[province]
            elif city != "":
                index_loc = city_map[city]
            else:
                if n_gram == 3:
                    return np.NaN
                n_gram += 1
            if index_loc == -1 and n_gram == 3:
                return np.NaN
            sz = len(json_city["features"][int(index_loc)]["geometry"]["coordinates"])
            sz2 = json_city["features"][int(index_loc)]["geometry"]["coordinates"][np.random.randint(sz)]
            loc = sz2[np.random.randint(len(sz2))]
            region = json_city["features"][int(index_loc)]["properties"]["REGION"]
            location = re.findall('\\d+.\\d+', str(loc))
            if loc is not None:
                found = True
        return str(float(location[1]) + -np.random.randint(2)) + "/" + str(
            float(location[0]) + 0.25) + ":" + init_loc + ":" + region

    no_coords['coords'] = no_coords['tweet'].apply(locate_tweet)

    # Add Names
    def add_location_name(text):
        return text.split(":")[1]

    def add_region(text):
        return text.split(":")[2]

    no_coords['loc_name'] = no_coords['coords'].apply(add_location_name)
    no_coords['region_name'] = no_coords['coords'].apply(add_region)
    # no_coords['region_id'] = no_coords['region_name'].apply(add_region_id)
    no_coords = no_coords.loc[no_coords['loc_name'] != ""]

    # Preprocess
    def preprocess_coords(coords):
        numbers = re.findall('\\d+', coords)
        return numbers[2] + "." + numbers[3] + "/" + numbers[0] + "." + numbers[1]

    no_coords['coords'] = no_coords['coords'].apply(preprocess_coords)
    no_coords.reset_index(inplace=True)

    return no_coords

def spatial_analysis_heat(data):
    specific_df = data.loc[data['type'] != ""].reset_index()[['coords', 'type']]

    heat_list = []
    for _loc in range(specific_df.shape[0]):
        # for _loc in range(10):
        temp = specific_df.coords[_loc].split("/")
        heat_list.append([float(temp[1]), float(temp[0]), 1])
    # heat_list = [[11.33877, 124.552177, 0.8]]

    map1 = folium.Map([12, 122], zoom_start=6)
    gradient = {'0.0': 'Navy', '0.25': 'Blue', '0.5': 'Green', '0.75': 'Yellow', '1': 'Red'}
    # HeatMap(data=heat_list, gradient=gradient, radius=25, blur = 10, min_opacity = 0.25, max_val = 0.0005).add_to(mapObj)
    plugins.HeatMap(heat_list, radius=25, max_zoom=13).add_to(map1)
    map1 = map1._repr_html_()
    return map1

def spatial_analysis_chloro(data):
    url = "https://raw.githubusercontent.com/lorensdima/datasetstest/main/Regions.json"
    f = requests.get(url)
    regions = f.json()

    def random_test(text):
        return np.random.randint(100)

    data["dummy"] = data["type"].apply(random_test)
    data

    fig = px.choropleth_mapbox(
        data,
        locations="region_name",
        geojson=regions,
        color="dummy",
        featureidkey="properties.REGION",
        hover_name="region_name",
        hover_data=["type"],
        mapbox_style="carto-positron",
        center={"lat": 12, "lon": 122},
        zoom=3,
    )
    return fig.to_html()
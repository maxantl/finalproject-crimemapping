from django.shortcuts import render
import folium
from folium import plugins
import plotly.express as px
import seaborn as sns
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
import pickle
warnings.filterwarnings('ignore')


def req(request):
    data = data_preparation()
    df = pd.DataFrame({'coords': data.coords, 'type': regression(data), 'region': data.region_name})
    final = df[df['type'] != 'non'].reset_index()
    map1 = spatial_analysis_heat(final)
    map2 = spatial_analysis_chloro(final)
    bar = create_bar(final)
    regression(data)
    context = {
        'map1': map1,
        'map2': map2,
        'title': "All Crimes",
        'bar': bar,
    }
    return render(request,'WEBAPP/notice.html',context)

def create_bar(df):
    temp = df[['type', 'region']]
    temp = temp.groupby(temp.columns.tolist(), as_index=False).size()
    fig = px.bar(temp, x="size", y="region", color="type", labels={
                     "size": "Crime Count",
                     "region": "Region Name",
                     "type": "Type of Crime",
                 },
                 title="Per Region Count")
    return fig.to_html()

def data_preparation():
    #url = "https://raw.githubusercontent.com/lorensdima/datasetstest/main/type_data.csv"
    url = "https://raw.githubusercontent.com/lorensdima/datasetstest/main/data16K.csv"
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
    no_coords = df[['tweet', 'coords']]

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


def change_type(request):
    data = data_preparation()
    df = pd.DataFrame({'coords': data.coords, 'type': regression(data), 'region': data.region_name})
    title = request.GET.get('dropdown_menu_form').upper()
    type = title.lower()
    specific_df = df[df['type'] != 'non'].reset_index()[['coords', 'type', 'region']]
    if type != "all":
        specific_df = specific_df[specific_df['type'] == type].reset_index()[['coords', 'type', 'region']]

    pred_df = pd.DataFrame({'type': specific_df.type, 'region': specific_df.region, 'coords': specific_df.coords})
    context = {
        'map1': spatial_analysis_heat(pred_df),
        'map2': spatial_analysis_chloro(pred_df),
        'title': title,
        'bar': create_bar(pred_df),
    }
    return render(request, 'WEBAPP/index.html', context)

def spatial_analysis_heat(data):
    specific_df = data[['coords', 'type']]
    heat_list = []
    for _loc in range(specific_df.shape[0]):
        # for _loc in range(10):
        temp = specific_df.coords[_loc].split("/")
        heat_list.append([float(temp[1]), float(temp[0]), 1])
    # heat_list = [[11.33877, 124.552177, 0.8]]

    map1 = folium.Map([12, 122], zoom_start=6, max_zoom=8)
    gradient = {'0.0': 'Navy', '0.25': 'Blue', '0.5': 'Green', '0.75': 'Yellow', '1': 'Red'}
    # HeatMap(data=heat_list, gradient=gradient, radius=25, blur = 10, min_opacity = 0.25, max_val = 0.0005).add_to(mapObj)
    plugins.HeatMap(heat_list, radius=25, max_zoom=13).add_to(map1)
    map1 = map1._repr_html_()
    return map1


def init_chloro(data):
    pred_df = pd.DataFrame({'tweet': data.tweet, 'type': regression(data), 'region': data.region_name})
    return spatial_analysis_chloro(pred_df)

def spatial_analysis_chloro(pred_df):
    regio = pred_df.region.unique()
    region_list = []
    type_list = []
    percent_list = []
    others_list = []
    for reg in regio:
        temp = pred_df[pred_df.region == reg].type.value_counts()
        region_list.append(reg)
        type_list.append(temp.index[0])
        percent_list.append(temp[0] / temp.sum() * 100)
        word = ""
        for item in temp.index:
            word += str(item) + ", "
        others_list.append(word)

    chloro_df = pd.DataFrame({'Region Name': region_list, 'Most Likely Crime': type_list,
                              'Likely %': percent_list, 'All Crimes': others_list})

    url = "https://raw.githubusercontent.com/lorensdima/datasetstest/main/Regions.json"
    f = requests.get(url)
    regions = f.json()

    fig = px.choropleth_mapbox(
        chloro_df,
        locations="Region Name",
        geojson=regions,
        color="Likely %",
        featureidkey="properties.REGION",
        hover_name="Region Name",
        hover_data=["All Crimes", "Most Likely Crime"],
        mapbox_style="carto-positron",
        center={"lat": 12, "lon": 122},
        zoom=5,
        height= 444,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
    )
    return fig.to_html()

def regression(data):
    df = data[['tweet']]
    # Download trained model
    url = "https://raw.githubusercontent.com/lorensdima/datasetstest/main/pickle_model.pkl"
    file = requests.get(url, stream=True)
    text_clf_svm = pickle.loads(file.content)
    final = text_clf_svm.predict(df.tweet)
    return final
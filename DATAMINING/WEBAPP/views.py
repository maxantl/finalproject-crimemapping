from django.shortcuts import render
import folium
from folium import plugins

def test(request):
    map1 = folium.Map([12, 122], tiles='CartoDB Dark_Matter', zoom_start=6, max_zoom=8)._repr_html_()
    context = {
        'map1': map1,
    }
    return render(request,'WEBAPP/test.html',context)
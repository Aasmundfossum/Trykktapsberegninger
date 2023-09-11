import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from scipy.spatial import ConvexHull
import math
import random
import requests
import folium
from folium import plugins
from streamlit_folium import st_folium
import leafmap.foliumap as leafmap
import geopandas

class Map:
    def __init__(self):
        self.address_lat = float
        self.address_long = float
        self.address_postcode = ""
        self.address_name = ""
        
        self.weather_station_lat = float
        self.weather_station_long = float
        self.weather_station_name = ""
        self.weather_station_id = ""
        
    def create_weather_station_map(self):
        st.markdown("---")
        st.header("Kart")
        selected_zoom = 13
        #--
        m = leafmap.Map(
            center=(self.address_lat, self.address_long), 
            zoom=selected_zoom,draw_control=False,
            measure_control=False,
            fullscreen_control=False,
            attribution_control=False,
            google_map="ROADMAP",
            shown=True
            )
        #--
        folium.Marker(
        [self.address_lat, self.address_long], 
        tooltip=f"{self.address_name}",
        icon=folium.Icon(icon="glyphicon-home", color="red"),
        ).add_to(m)
        #--
        folium.Marker(
        [self.weather_station_lat, self.weather_station_long], 
        tooltip=f"""ID: {self.weather_station_id} <br>Navn: {self.weather_station_name} """,
        icon=folium.Icon(icon="glyphicon-cloud", color="blue"),
        ).add_to(m)
        #--
        self.m = m
        
    def _draw_polygon(self):
        plugins.Draw(
        export=False,
        position="topleft",
        draw_options={
            "polyline": False,
            "poly": False,
            "circle": False,
            "polygon": True,
            "marker": False,
            "circlemarker": False,
            "rectangle": False,
        },
        ).add_to(self.m)
 
    def create_wms_map(self, selected_display = True):
        if selected_display == True:
            st.markdown("---")
            st.header("Kart")
            selected_display = st.radio("Visningsalternativer", ["Oversiktskart", "Løsmasserelatert", "Berggrunnsrelatert"])
        selected_zoom = 13
        #--
        m = leafmap.Map(
            center=(self.address_lat, self.address_long), 
            zoom=selected_zoom,
            draw_control=False,
            measure_control=False,
            fullscreen_control=False,
            attribution_control=False,
            google_map="ROADMAP",
            shown=True
            )
        #--
        folium.Marker(
        [self.address_lat, self.address_long], 
        tooltip=f"{self.address_name}",
        icon=folium.Icon(icon="glyphicon-home", color="red"),
        ).add_to(m)
        #--
        wms_url_list = [
            "https://geo.ngu.no/mapserver/LosmasserWMS?request=GetCapabilities&service=WMS",
            "https://geo.ngu.no/mapserver/MarinGrenseWMS4?REQUEST=GetCapabilities&SERVICE=WMS",
            "https://geo.ngu.no/mapserver/GranadaWMS5?request=GetCapabilities&service=WMS",
            "https://geo.ngu.no/geoserver/nadag/ows?request=GetCapabilities&service=WMS",
            "https://geo.ngu.no/mapserver/BerggrunnWMS3?request=GetCapabilities&SERVICE=WMS",
            "https://geo.ngu.no/mapserver/BerggrunnWMS3?request=GetCapabilities&SERVICE=WMS",
            "https://geo.ngu.no/mapserver/BerggrunnWMS3?request=GetCapabilities&SERVICE=WMS",
            
        ]
        wms_layer_list = [
            "Losmasse_flate",
            "Marin_grense_linjer",
            "Energibronn",
            "GBU_metode",
            "Berggrunn_lokal_hovedbergarter",
            "Berggrunn_regional_hovedbergarter",
            "Berggrunn_nasjonal_hovedbergarter",
        ]
        wms_name_list = [
            "Løsmasser",
            "Marin grense",            
            "Energibrønner",
            "Grunnundersøkelser",
            "Lokal berggrunn",
            "Regional berggrunn",
            "Nasjonal berggrunn",
        ]
        for i in range(0, len(wms_url_list)):
            display = False
            if selected_display == "Løsmasserelatert" and i < 4:
                display = True 
            if selected_display == "Berggrunnsrelatert" and i == 4:
                display = True
            self._add_wms_layer(
                m,
                wms_url_list[i],
                wms_layer_list[i],
                wms_name_list[i],
                display
            )
        self.m = m
    
    def show_map(self):
        self.m.to_streamlit(700, 600)
        
    def _add_wms_layer(self, map, url, layer, layer_name, display):
        map.add_wms_layer(
            url, 
            layers=layer, 
            name=layer_name, 
            attribution=" ", 
            transparent=True,
            format="image/png",
            shown=display
            )
    
    def _style_function(self, x):
        return {"color":"black", "weight":2}

    def _add_geojson_layer(self, filepath, layer_name):
        uc = "\u00B2"
        buildings_gdf = geopandas.read_file(filepath)
        buildings_df = buildings_gdf[['ID', 'BRA', 'Kategori', 'Standard']]
        #folium.GeoJson(data=buildings_gdf["geometry"]).add_to(m)

        feature = folium.features.GeoJson(buildings_gdf,
        name=layer_name,
        style_function=self._style_function,
        tooltip=folium.GeoJsonTooltip(fields= ["ID", "BRA"],aliases=["ID: ", f"BTA (m{uc}): "],labels=True))
        self.m.add_child(feature)
    
    def create_map_old(self):
        st.subheader("Oversiktskart")
        m = folium.Map(
            location=[self.address_lat, self.address_long], 
            zoom_start=12, 
            zoom_control=True, 
            dragging=True,
            scrollWheelZoom=True,
            tiles="OpenStreetMap", 
            no_touch=True, 
            )
        folium.Marker(
            [self.address_lat, self.address_long], 
            tooltip=f"{self.address_name}",
            icon=folium.Icon(icon="glyphicon-home", color="red"),
        ).add_to(m)

        folium.Marker(
            [self.weather_station_lat, self.weather_station_long], 
            tooltip=f"""ID: {self.weather_station_id} <br>Navn: {self.weather_station_name} <br>Avstand: {self.weather_station_distance} km""",
            icon=folium.Icon(icon="glyphicon-cloud", color="blue"),
        ).add_to(m)

        selected_url = 'https://geo.ngu.no/mapserver/LosmasserWMS?request=GetCapabilities&service=WMS'
        selected_layer = 'Losmasse_flate'

        folium.raster_layers.WmsTileLayer(url = selected_url,
            layers = selected_layer,
            transparent = True, 
            control = True,
            fmt="image/png",
            name = 'Løsmasser',
            overlay = True,
            show = False,
            CRS = 'EPSG:900913',
            version = '1.3.0',
            ).add_to(m)

        folium.LayerControl(position = 'bottomleft').add_to(m)
        st_folium(m, width = 700)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------




st.set_page_config(
    page_title="Grunnvarme",
    page_icon="🏔️",
)
#with open("styles/main.css") as f:
#    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

def plot_eb(x, y, id = 1):
    plt.scatter(x, y, color = "blue", marker = ".")
    plt.annotate(id, (x, y))
    circle1 = plt.Circle((x, y), 15, linestyle = "--", fill = None, alpha = 0.1)
    plt.gca().add_patch(circle1)

def plot_sk(x, y, id = 1):
    plt.scatter(x, y, color = "red", marker = ".")
    plt.annotate(id, (x, y))

def plot_tr(x, y, id = 1):
    plt.scatter(x, y, color = "green", marker = ".")
    plt.annotate(id, (x, y))

def plot_point(x, y, id = 1):
    id_type = id[0:2].upper()
    if id_type == "EB":
        plot_eb(x, y, id)
    elif id_type == "SK":
        plot_sk(x, y, id)
    elif id_type == "TR":
        plot_tr(x, y, id)

def plot_line(point1 = [1, 2], point2 = [3, 4], line_length = 5, line_color = "black", line_alpha = 0.2):
    x_values = [point1[0], point2[0]]
    y_values = [point1[1], point2[1]]
    plt.plot(x_values, y_values, linestyle="-", color = line_color, alpha = line_alpha)
    pos_y1 = center_point(x_values[0], x_values[1], y_values[0], y_values[1])
    #plt.plot(pos_y1[0], pos_y1[1], color = "green", marker = ".")
    plt.annotate(f'{line_length}', xy =(0, 0), xytext =(pos_y1[0], pos_y1[1]))
  
def calculate_line_length(point1 = [1, 2], point2 = [3, 4]):
    x2, y2 = point2[0], point2[1]
    x1, y1 = point1[0], point1[1]
    length = abs((((x2-x1)**2) + ((y2-y1)**2))**(1/2))
    return int(round(length, 0))

def center_point(x1, x2, y1, y2):
    x_mid = ((x2 + x1))/2
    y_mid = ((y2 + y1))/2
    return x_mid, y_mid

def set_cx_cy(points_SK, points_EB):
    cx = points_SK[0][0]
    cy = points_SK[0][1]
    total_line_length = 0
    max_line_length = 0
    for i in range(0, len(points_EB)):
        line_length = calculate_line_length(point1 = [cx, cy], point2 = points_EB[i])
        if line_length > max_line_length:
            max_line_length = line_length
        total_line_length += line_length
    return cx, cy, total_line_length, max_line_length

def find_cx_cy(points, mode = "Iterasjon"):
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], color = "red", linestyle = "--", alpha = 0.2)
    if mode == "Tyngdepunkt":
        cx = np.mean(hull.points[hull.vertices,0])
        cy = np.mean(hull.points[hull.vertices,1])
        total_line_length = 0
        max_line_length = 0
        for i in range(0, len(points)):
            line_length = calculate_line_length(point1 = [cx, cy], point2 = points[i])
            if line_length > max_line_length:
                max_line_length = line_length
            total_line_length += line_length
    elif mode == "Iterasjon":
        x_points = hull.points[hull.vertices,0]
        y_points = hull.points[hull.vertices,1]
        p1 = [int(np.min(x_points)), int(np.min(y_points))]
        p2 = [int(np.max(x_points)), int(np.max(y_points))]
        MLL, TLL, CX, CY = [], [], [], []
        for x in range(p1[0], p2[0]):
            for y in range(p1[0], p2[1]):
                cx, cy = x, y
                #--
                #ax.plot(cx, cy, marker = "o", color = "red")
                total_line_length = 0
                max_line_length = 0
                for i in range(0, len(points)):
                    line_length = calculate_line_length(point1 = [cx, cy], point2 = points[i])
                    if line_length > max_line_length:
                        max_line_length = line_length
                    total_line_length += line_length
                MLL.append(max_line_length)
                TLL.append(total_line_length)
                CX.append(cx)
                CY.append(cy)
        index = np.argmin(MLL)
        cx = CX[index]
        cy = CY[index]
        total_line_length = TLL[index]
        max_line_length = MLL[index]
    return cx, cy, total_line_length, max_line_length

def filter_dataframe(id):
    return df[df["ID"].str.contains(id)]

def add_random_eb(dataframe):
    df_copy = dataframe.copy()  # Create a copy of the dataframe to avoid modifying the original
    random_eb = f"EB{random.randint(1, 100)}"  # Generate a random EB value
    df_copy.loc[df_copy.shape[0]] = [random_eb, random.randint(-100, 100), random.randint(-100, 100)]  # Add the random EB row to the dataframe
    return df_copy

#------------

st.button("Oppdater")
st.title("Brønnkonfigurasjoner")

#tab1, tab2, tab3 = st.tabs(["40 mm", "45 mm", "50 mm"])

df = pd.DataFrame({
    "ID" : ["EB1", "EB2", "EB3", "EB4", "EB5", "EB6", "EB7", "TR", "SK1", "EB8"],
    "X" : [5, 35, 20, -10, -25, 20, -10, 50, 0, 60],
    "Y" : [5, 5, 31, 31, 5, -21, -21, 20, 50, 60]})

fig, ax = plt.subplots()
for i in range(0, len(df)):
    X, Y, ID = df["X"][i], df["Y"][i], df["ID"][i]
    plot_point(X, Y, ID)

df_EB, df_TR, df_SK = filter_dataframe("EB"), filter_dataframe("TR"), filter_dataframe("SK")
points_EB, points_TR, points_SK = df_EB[["X", "Y"]].to_numpy(dtype=float), df_TR[["X", "Y"]].to_numpy(dtype=float), df_SK[["X", "Y"]].to_numpy(dtype=float)

selected_mode = st.selectbox("Modus", options=["Iterasjon", "Tyngdepunkt", "Egendefinert"])

if selected_mode == "Egendefinert":
    cx, cy, total_line_length, max_line_length = set_cx_cy(points_SK, points_EB)
else:
    cx, cy, total_line_length, max_line_length = find_cx_cy(points_EB, mode = selected_mode)

ax.plot(cx, cy, marker = "o", color = "red")
for i in range(0, len(points_EB)):
    line_length = calculate_line_length(point1 = [cx, cy], point2 = points_EB[i])
    plot_line(point1 = [cx, cy], point2 = points_EB[i], line_length = line_length)
if len(points_TR) > 0:
    line_length = calculate_line_length(point1 = [cx, cy], point2 = [points_TR[0][0], points_TR[0][1]])  
    sk_to_tr_length = line_length  
    plot_line(point1 = [cx, cy], point2 = [points_TR[0][0], points_TR[0][1]], line_length = line_length, line_alpha = 0.5)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Maks (SK <-> EB)", f"{max_line_length} m")
with c2:
    st.metric("Totalt (SK <-> EB)", f"{total_line_length} m")
with c3:
    st.metric("SK <-> TR", f"{sk_to_tr_length} m")

plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.grid(True)
plt.gca().set_aspect("equal")
st.pyplot(plt)
plt.close()

# ------------------------------------------Mitt rot -----------------------------------------
def egenskap_funk(fryspunkt, vaesketemp, D_values):
        xm = D_values.iloc[-2]
        ym = D_values.iloc[-1]
        
        result = (
            D_values.iloc[0]
            + D_values.iloc[1] * (vaesketemp - ym)
            + D_values.iloc[2] * (vaesketemp - ym) ** 2
            + D_values.iloc[3] * (vaesketemp - ym) ** 3
            + D_values.iloc[4] * (fryspunkt - xm)
            + D_values.iloc[5] * (fryspunkt - xm) * (vaesketemp - ym)
            + D_values.iloc[6] * (fryspunkt - xm) * (vaesketemp - ym) ** 2
            + D_values.iloc[7] * (fryspunkt - xm) * (vaesketemp - ym) ** 3
            + D_values.iloc[8] * (fryspunkt - xm) ** 2
            + D_values.iloc[9] * (fryspunkt - xm) ** 2 * (vaesketemp - ym)
            + D_values.iloc[10] * (fryspunkt - xm) ** 2 * (vaesketemp - ym) ** 2
            + D_values.iloc[11] * (fryspunkt - xm) ** 2 * (vaesketemp - ym) ** 3
            + D_values.iloc[12] * (fryspunkt - xm) ** 3
            + D_values.iloc[13] * (fryspunkt - xm) ** 3 * (vaesketemp - ym)
            + D_values.iloc[14] * (fryspunkt - xm) ** 3 * (vaesketemp - ym) ** 2
            + D_values.iloc[15] * (fryspunkt - xm) ** 4
            + D_values.iloc[16] * (fryspunkt - xm) ** 4 * (vaesketemp - ym)
            + D_values.iloc[17] * (fryspunkt - xm) ** 5
        )
        return result

def trykktapsfunksjon(volstrom,tetthet,diam_ytre,veggtykkelse,viskositet,ruhet,lengde):
    massestrom = volstrom*tetthet/1000
    diam_indre = (diam_ytre-2*veggtykkelse)/1000
    Re = (4*massestrom)/(viskositet*np.pi*diam_indre)
    frikfaktor = (1/(-1.8*np.log10(6.9/Re)+((ruhet/diam_indre)/3.7)**1.11))**2
    hast = massestrom/(tetthet*np.pi*(diam_indre/2)**2)
    trykktap = frikfaktor*lengde/diam_indre*(tetthet*hast**2)/2
    return Re,trykktap

#def trykktapsfunksjon_VP(volstrom,tetthet,diam_ytre,veggtykkelse,viskositet,lengde):
#    massestrom = volstrom*tetthet/1000
#    diam_indre = (diam_ytre-2*veggtykkelse)/1000
#    Re = (4*massestrom)/(viskositet*np.pi*diam_indre)
#    frikfaktor = 64/Re
#    hast = massestrom/(tetthet*np.pi*(diam_indre/2)**2)
#    trykktap = frikfaktor*lengde/diam_indre*(tetthet*hast**2)/2
#    return trykktap

def Nusselt(Re,Pr):
    teller = (1/(0.79*np.log(Re)-1.64)**2)/(8)*(Re-1000)*Pr
    nevner = 1+12.7*((1/(0.79*np.log(Re)-1.64)**2)/(8))**(1/2)*Pr**(2/3)-1
    Nu = teller/nevner                                                      #Kun hvis Re > 2300 
    return Nu

def trykktap_mm_funksjon(max_line_length,sk_to_tr_length):
    # Input:    Lengden til det lengste rørstrekket fra brønn til samlekum
    #           Lengden på hovedtraseen fra samlekum til termisk rom
    # Output:   Streamlit-nettside hvor diverse input kan velges og resultater skrives ut
    
    st.title("Trykktap")
    st.subheader('Input:')
    d1, d2 = st.columns(2)
    
    #kollvaeske = pd.read_excel("Kollektorvæskedata.xlsx", sheet_name="Sheet1")
    #with d1:
    #    vaeske = st.selectbox("Kollektorvæske:", options=['Vann','Monoetylenglykol 25 %','Monoetylenglykol 33 %','Monopropylenglykol 25 %','Monopropylenglykol 33 %','Metanol 25 %','Etanol 15 %',
    #                                                'Etanol 20 %','Etanol 24 %','Etanol 28 %','Etanol 35 %','Kaliumkarbonat 25 %','Kaliumkarbonat 33 %','Kalsiumklorid 20 %',
    #                                                'Kilfrost 24 %'], index=8)
    #riktig_rad = kollvaeske[kollvaeske['Cooling liquid'].str.contains(vaeske)] 
    #with d2:
    #    vaesketemperatur = st.selectbox("Kollektorvæsketemperatur (\u2103):", options=riktig_rad.iloc[:,1])
    #riktig_rad = riktig_rad.loc[riktig_rad['temp (C)']==vaesketemperatur]

        #k = riktig_rad.iloc[0,2]
    #cp = riktig_rad.iloc[0,3]
    #tetthet = riktig_rad.iloc[0,4]
    #viskositet = riktig_rad.iloc[0,5]
    #Pr = riktig_rad.iloc[0,7]

    # Valg av kjølevæske og beregning av egenskaper til denne;
    vaeske = st.selectbox("Kollektorvæske:", options=['Etylenglykol','Propylenglykol','Etylalkohol','Metylalkohol','Glycerin','Ammoniak','Kaliumkarbonat',
                                                    'Kalciumklorid','Magnesiumklorid','Natriumklorid','Kaliumacetat'], index=0)

    if vaeske =='Etylenglykol':
        min_frystemp = -45
        max_frystemp = 0
        max_temp = 40
    elif vaeske =='Propylenglykol':
        min_frystemp = -45
        max_frystemp = -5
        max_temp = 40
    elif vaeske =='Etylalkohol':
        min_frystemp = -45
        max_frystemp = -5
        max_temp = 20
    elif vaeske =='Metylalkohol':
        min_frystemp = -50
        max_frystemp = -5
        max_temp = 20
    elif vaeske =='Glycerin':
        min_frystemp = -40
        max_frystemp = -5
        max_temp = 40
    elif vaeske =='Ammoniak':
        min_frystemp = -50
        max_frystemp = -10
        max_temp = 20
    elif vaeske =='Kaliumkarbonat':
        min_frystemp = -35
        max_frystemp = -0
        max_temp = 30
    elif vaeske =='Kalciumklorid':
        min_frystemp = -45
        max_frystemp = -5
        max_temp = 30
    elif vaeske =='Magnesiumklorid':
        min_frystemp = -30
        max_frystemp = -0
        max_temp = 30
    elif vaeske =='Natriumklorid':
        min_frystemp = -20.7
        max_frystemp = -0
        max_temp = 30
    elif vaeske =='Kaliumacetat':
        min_frystemp = -45
        max_frystemp = -5
        max_temp = 30

    c1, c2 = st.columns(2)
    with c1:
        fryspunkt = st.number_input(label='Frysepunkttemperatur (mellom '+str(min_frystemp)+' \u2103 og '+str(max_frystemp)+' \u2103)', min_value=float(min_frystemp), max_value=float(max_frystemp), value=float(-18), step=0.1)
    with c2:
        vaesketemp = st.number_input(label='Kjølevæsketemperatur (mellom '+str(round(fryspunkt,1))+' \u2103 og '+str(max_temp)+' \u2103)', min_value=fryspunkt, max_value=float(max_temp), value=float(-2), step=0.1)

    all_values = pd.read_excel("Komplett_datablad.xlsx", sheet_name=vaeske)             # Leser av arket som samsvarer med den valgte kollektorvæsken
    til_konsentrasjon = all_values.iloc[:,3]
    til_tetthet = all_values.iloc[:,4]
    til_varmekap = all_values.iloc[:,5]
    til_ledningsevne = all_values.iloc[:,6]
    til_viskositet = all_values.iloc[:,7]

    konsentrasjon = egenskap_funk(fryspunkt, vaesketemp, til_konsentrasjon)         # %
    tetthet = egenskap_funk(fryspunkt, vaesketemp, til_tetthet)
    varmekap = egenskap_funk(fryspunkt, vaesketemp, til_varmekap)                   # J/(kg K)
    ledningsevne = egenskap_funk(fryspunkt, vaesketemp, til_ledningsevne)           # W/(m K)
    viskositet = np.exp(egenskap_funk(fryspunkt, vaesketemp, til_viskositet))/1000   #Pa*s = kg/(m s)
    Pr = viskositet*varmekap/ledningsevne

    # Valg av volumstrøm og brønnlengde
    d1, d2 = st.columns(2)
    with d1:
        volstrom = st.number_input(label='Volumstrøm (l/s)', min_value=0.01, value=0.69, step=0.01)
    with d2:
        lengde_bronn = st.number_input(label='Brønnlengde tur/retur (m)', min_value=1, value=600, step=1)

    # Valg av trykkenhet
    trykkenhet = st.selectbox('Enhet for trykk', options=['Pascal (Pa)','Megapascal (MPa)', 'Bar (bar)', 'Pund per kvadrattomme (psi)'], index=0)
    c1, c2, c3 = st.columns(3)
    if trykkenhet == 'Pascal (Pa)':
        with c1:
            trykktapsgrense_bronn = st.number_input(label='Maksimalt tillatt trykktap inni brønn (Pa)', min_value=1, value=10000, step=1000)
        with c2:
            trykktapsgrense_rorstrekk = st.number_input(label='Maksimalt tillatt trykktap i rørstrekk fra brønn (Pa)', min_value=1, value=10000, step=1000)
        with c3:
            trykktapsgrense_hovedtrase = st.number_input(label='Maksimalt tillatt trykktap i hovedtrase fra samlekum (Pa)', min_value=1, value=10000, step=1000)
    elif trykkenhet == 'Megapascal (MPa)':
        with c1:
            trykktapsgrense_bronn = (10**6)*st.number_input(label='Maksimalt tillatt trykktap inni brønn (MPa)', min_value=0.000001, value=0.01, step=0.001)
        with c2:
            trykktapsgrense_rorstrekk = (10**6)*st.number_input(label='Maksimalt tillatt trykktap i rørstrekk fra brønn (MPa)', min_value=0.000001, value=0.01, step=0.001)
        with c3:
            trykktapsgrense_hovedtrase = (10**6)*st.number_input(label='Maksimalt tillatt trykktap i hovedtrase fra samlekum (MPa)', min_value=0.000001, value=0.01, step=0.001)
    elif trykkenhet == 'Bar (bar)':
        with c1:
            trykktapsgrense_bronn = (10**5)*st.number_input(label='Maksimalt tillatt trykktap inni brønn (bar)', min_value=0.00001, value=0.1, step=0.01)
        with c2:
            trykktapsgrense_rorstrekk = (10**5)*st.number_input(label='Maksimalt tillatt trykktap i rørstrekk fra brønn (bar)', min_value=0.00001, value=0.1, step=0.01)
        with c3:
            trykktapsgrense_hovedtrase = (10**5)*st.number_input(label='Maksimalt tillatt trykktap i hovedtrase fra samlekum (bar)', min_value=0.00001, value=0.1, step=0.01)
    elif trykkenhet == 'Pund per kvadrattomme (psi)':
        with c1:
            trykktapsgrense_bronn = (6894.75729)*st.number_input(label='Maksimalt tillatt trykktap inni brønn (psi)', min_value=0.0001450377, value=1.4503773773, step=0.1450377377)
        with c2:
            trykktapsgrense_rorstrekk = (6894.75729)*st.number_input(label='Maksimalt tillatt trykktap i rørstrekk fra brønn (psi)', min_value=0.0001450377, value=1.4503773773, step=0.1450377377)
        with c3:
            trykktapsgrense_hovedtrase = (6894.75729)*st.number_input(label='Maksimalt tillatt trykktap i hovedtrase fra samlekum (psi)', min_value=0.0001450377, value=1.4503773773, step=0.1450377377)

    # Valg av rør:
    antall_bronner = len(df)-2
    ruhet = 0.0000015
    rordata = np.array([[32,2.0],[40,2.4],[50,3.0],[63,3.8],[75,4.5],[90,5.4],[110,6.6],[125,7.4],[140,8.3],[160,9.5],[180,10.7],[200,11.9],[225,13.4],[250,14.8],[280,16.6],
                        [315,18.7],[355,21.1],[400,23.7],[450,26.7],[500,29.7],[560,33.2],[600,35.6],[630,37.4],[710,42.1],[800,47.4],[900,53.3],[1000,59.3],[1100,65.2],[1200,70.6],
                        [1400,82.4],[1600,94.1],[1800,105.9],[2000,117.6]])   # SDR 17

    for i in range(0,len(rordata)):
        veggtykkelse_bronn = rordata[i,1]    # mm
        diam_ytre_bronn = rordata[i,0]        # mm
        [Re_bronn, trykktap_bronn] = trykktapsfunksjon(volstrom,tetthet,diam_ytre_bronn,veggtykkelse_bronn,viskositet,ruhet,lengde_bronn)
        if trykktap_bronn <= trykktapsgrense_bronn:
            break

    for i in range(0,len(rordata)):
        veggtykkelse_rorstrekk = rordata[i,1]    # mm
        diam_ytre_rorstrekk = rordata[i,0]        # mm
        [Re_rorstrekk, trykktap_rorstrekk] = trykktapsfunksjon(volstrom,tetthet,diam_ytre_rorstrekk,veggtykkelse_rorstrekk,viskositet,ruhet,max_line_length*2)
        if trykktap_rorstrekk <= trykktapsgrense_rorstrekk:
            break

    for i in range(0,len(rordata)):
        veggtykkelse_hovedtrase = rordata[i,1]    # mm
        diam_ytre_hovedtrase = rordata[i,0]        # mm
        [Re_hovedtrase, trykktap_hovedtrase] = trykktapsfunksjon(volstrom*antall_bronner,tetthet,diam_ytre_hovedtrase,veggtykkelse_hovedtrase,viskositet,ruhet,sk_to_tr_length*2)
        if trykktap_hovedtrase <= trykktapsgrense_hovedtrase:
            break

    #trykktap_VP = trykktapsfunksjon_VP(volstrom_hovedtrase,tetthet,diam_ytre_hovedtrase,veggtykkelse_hovedtrase,viskositet,lengde_hovedtrase)
    trykktap_tot = trykktap_bronn+trykktap_rorstrekk+trykktap_hovedtrase # +trykktap_VP

    #### Skriver ut alle resultater:
    st.header('Resultater:')
    st.subheader('Konsentrasjon til kjølevæske:')
    st.metric('Nødvendig konsentrasjon for å oppnå ønsket frysepunkttemperatur:',f'{round(konsentrasjon,2)} %')

    st.subheader('Ytre rørdiameter ($D$):')
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Energibrønn", f"{diam_ytre_bronn} mm")
    with c2:
        st.metric("Rørstrekk fra brønn til samlekum", f"{diam_ytre_rorstrekk} mm")
    with c3:
        st.metric("Hovedtrasé fra samlekum til TR", f"{diam_ytre_hovedtrase} mm")

    st.subheader('Trykktap ($\Delta P$):')
    c1, c2, c3 = st.columns(3)
    if trykkenhet == 'Pascal (Pa)':
        with c1:
            st.metric("Energibrønn", f"{round(trykktap_bronn)} Pa")
        with c2:
            st.metric("Rørstrekk fra brønn til samlekum", f"{round(trykktap_rorstrekk)} Pa")
        with c3:
            st.metric("Hovedtrasé fra samlekum til TR", f"{round(trykktap_hovedtrase)} Pa")
        st.metric("Totalt:", f"{round(trykktap_tot)} Pa")
    elif trykkenhet == 'Megapascal (MPa)':
        with c1:
            st.metric("Energibrønn", f"{round(trykktap_bronn/(10**6),3)} MPa")
        with c2:
            st.metric("Rørstrekk fra brønn til samlekum", f"{round(trykktap_rorstrekk/(10**6),4)} MPa")
        with c3:
            st.metric("Hovedtrasé fra samlekum til TR", f"{round(trykktap_hovedtrase/(10**6),4)} MPa")
        st.metric("Totalt:", f"{round(trykktap_tot/(10**6),4)} MPa")
    elif trykkenhet == 'Bar (bar)':
        with c1:
            st.metric("Energibrønn", f"{round(trykktap_bronn/(10**5),3)} bar")
        with c2:
            st.metric("Rørstrekk fra brønn til samlekum", f"{round(trykktap_rorstrekk/(10**5),3)} bar")
        with c3:
            st.metric("Hovedtrasé fra samlekum til TR", f"{round(trykktap_hovedtrase/(10**5),3)} bar")
        st.metric("Totalt:", f"{round(trykktap_tot/(10**5),3)} bar")
    elif trykkenhet == 'Pund per kvadrattomme (psi)':
        with c1:
            st.metric("Energibrønn", f"{round(trykktap_bronn/(6894.75729),3)} psi")
        with c2:
            st.metric("Rørstrekk fra brønn til samlekum", f"{round(trykktap_rorstrekk/(6894.75729),3)} psi")
        with c3:
            st.metric("Hovedtrasé fra samlekum til TR", f"{round(trykktap_hovedtrase/(6894.75729),3)} psi")
        st.metric("Totalt:", f"{round(trykktap_tot/(6894.75729),3)} psi")

    st.subheader('Reynolds-tall ($Re$):')
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Energibrønn", f"{round(Re_bronn,2)}")
    with c2:
        st.metric("Rørstrekk fra brønn til samlekum", f"{round(Re_rorstrekk,2)}")
    with c3:
        st.metric("Hovedtrasé fra samlekum til TR", f"{round(Re_hovedtrase,2)}")

    #Varmetap:
    Nu_bronn = Nusselt(Re_bronn,Pr)
    h_bronn = (Nu_bronn*ledningsevne)/((diam_ytre_bronn-2*veggtykkelse_bronn)/1000)
    R_konv_bronn = 1/(np.pi*((diam_ytre_bronn-2*veggtykkelse_bronn)/1000)*h_bronn)
    R_ror_bronn = 1/(2*np.pi*0.42)*np.log(0.02/0.0176)

    Nu_rorstrekk = Nusselt(Re_rorstrekk,Pr)
    h_rorstrekk = (Nu_rorstrekk*ledningsevne)/((diam_ytre_rorstrekk-2*veggtykkelse_rorstrekk)/1000)
    R_konv_rorstrekk = 1/(np.pi*((diam_ytre_rorstrekk-2*veggtykkelse_rorstrekk)/1000)*h_rorstrekk)

    Nu_hovedtrase = Nusselt(Re_hovedtrase,Pr)
    h_hovedtrase = (Nu_hovedtrase*ledningsevne)/((diam_ytre_hovedtrase-2*veggtykkelse_hovedtrase)/1000)
    R_konv_hovedtrase = 1/(np.pi*((diam_ytre_hovedtrase-2*veggtykkelse_hovedtrase)/1000)*h_hovedtrase)

    st.title("Varmetap")
    st.subheader('Nusselt-tall ($Nu$):')
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Energibrønn", f"{round(Nu_bronn,2)}")
    with c2:
        st.metric("Rørstrekk fra brønn til samlekum", f"{round(Nu_rorstrekk,2)}")
    with c3:
        st.metric("Hovedtrasé fra samlekum til TR", f"{round(Nu_hovedtrase,2)}")

    st.subheader('Konveksjonsvarmeoverføringskoeffisient ($h$)')
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Energibrønn", f"{round(h_bronn,1)} W/m\u00b2K")
    with c2:
        st.metric("Rørstrekk fra brønn til samlekum", f"{round(h_rorstrekk,1)} W/m\u00b2K")
    with c3:
        st.metric("Hovedtrasé fra samlekum til TR", f"{round(h_hovedtrase,1)} W/m\u00b2K")

    st.subheader('Konduksjonsvarmemotstand i rør ($R_{kond}$)')
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Energibrønn", f"{round(R_konv_bronn,3)} K/W")
    with c2:
        st.metric("Rørstrekk fra brønn til samlekum", f"{round(R_konv_rorstrekk,3)} K/W")
    with c3:
        st.metric("Hovedtrasé fra samlekum til TR", f"{round(R_konv_hovedtrase,3)} K/W")


#Bruker funksjonen:
trykktap_mm_funksjon(max_line_length,sk_to_tr_length)
import streamlit as st
import pandas as pd
import numpy as np

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


st.set_page_config(
    page_title="Trykktap",
    page_icon="🏔️",
)

st.title("Trykktapsberegning for energibrønn")

st.markdown('---')
st.subheader('Input:')

feil = False
vaeske = st.selectbox("Kollektorvæske:", options=['Egendefinerte kjølevæskeegenskaper','Etylenglykol','Propylenglykol','Etylalkohol','Metylalkohol','Glycerin','Ammoniak','Kaliumkarbonat',
                                                        'Kalciumklorid','Magnesiumklorid','Natriumklorid','Kaliumacetat'], index=0)
if vaeske != 'Egendefinerte kjølevæskeegenskaper':

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


elif vaeske == 'Egendefinerte kjølevæskeegenskaper':
    c1,c2 = st.columns(2)
    with c1:
        tetthet=st.number_input('Tetthet til kjølevæske (kg/m$^3$)',value=float(1000),step=0.1)
    with c2:
        ledningsevne=st.number_input('Ledningsevne til kjølevæske (W/mK)',value=0.5,step=0.01)
    c1,c2 = st.columns(2)
    with c1:
        viskositet=st.number_input('Viskositet til kjølevæske (Pa s)',value=0.0042,step=0.0001,format="%f")
    with c2:
        Pr=st.number_input('Prandtl-tall til kjølevæske (-)',value=float(34),step=0.1)

rortabell = pd.read_excel("SDR-tabell.xlsx", sheet_name='Sheet1')
diam_liste = rortabell.iloc[1:,0]
SDR_liste_hele = rortabell.columns.values.tolist()
SDR_liste = []
for i in range(0,len(SDR_liste_hele)):
    if 'SDR' in SDR_liste_hele[i]:
        SDR_liste.append(SDR_liste_hele[i])

d1, d2, d3 = st.columns(3)
with d1:
    koll_diam = st.number_input(label='Kollektordiameter (mm)', value = 50)
with d2:
    sdr = st.selectbox(label='SDR', options=SDR_liste)
with d3:
    ruhet_mm = st.number_input(label='Ruhet (mm)',value = 0.0015,step=0.1,format="%f")

ror_knapp = st.checkbox('Velg et tilgjengelig rør fra tabell som brukes i videre beregninger')

ruhet = ruhet_mm/1000
sdr_tall = float(sdr.replace('SDR ',''))
indre_diam_faktisk = (koll_diam-2*koll_diam/sdr_tall)
tykk_faktisk = indre_diam_faktisk/sdr_tall
indre_diam=indre_diam_faktisk

if ror_knapp == True:
    stopp = False
    for i in range(1,len(rortabell.iloc[1:,0])):
        if indre_diam_faktisk <= rortabell.iloc[i,0]:
            riktig_rad = i-1
            break

    for i in range(0,len(SDR_liste)):
        if sdr == SDR_liste[i]:
            tykk_kol = rortabell.iloc[1:,2*i+1]
            vekt_kol = rortabell.iloc[1:,2*i+2]
            print(tykk_kol)
            break


    tykk_tabell = tykk_kol.iloc[riktig_rad]
    rorvekt = vekt_kol.iloc[riktig_rad]

    if tykk_tabell != tykk_tabell:
        st.markdown(f'Rørdiameter rundt {round(indre_diam_faktisk,1)} mm er ikke tilgjengelig for {sdr}. Prøv å endre kollektordiameter og/eller SDR.')
        feil = True
 
    else:
        indre_diam_tabell = rortabell.iloc[riktig_rad+1,0]
        indre_diam = indre_diam_tabell


d1, d2 = st.columns(2)
with d1:
    volstrom = st.number_input(label='Volumstrøm (l/s)', min_value=0.01, value=0.7, step=0.01)
with d2:
    dybde_bronn = st.number_input(label='Brønndybde (m)', min_value=1, value=300, step=1)

lengde_bronn = 2*dybde_bronn

massestrom = volstrom*tetthet/1000

trykkenhet = st.selectbox('Enhet for trykk', options=['Pascal (Pa)','Megapascal (MPa)', 'Bar (bar)', 'Pund per kvadrattomme (psi)'], index=0)

Re = (massestrom*indre_diam/1000)/(viskositet*np.pi*(indre_diam/1000/2)**2)
frikfaktor = (1/(-1.8*np.log10(6.9/Re)+((ruhet/indre_diam/1000)/3.7)**1.11))**2
hast = massestrom/(tetthet*np.pi*(indre_diam/1000/2)**2)
trykktap = frikfaktor*lengde_bronn/indre_diam*1000*(tetthet*hast**2)/2


c1, c2, c3, c4, c5 = st.columns(5)
with c3:
    resultat_knapp = st.button('Vis resultater')


if resultat_knapp == True and feil == True:
    st.markdown('Rett opp feilen og prøv igjen')
elif resultat_knapp == True and feil == False:
    st.markdown('---')
    st.subheader('Diverse parametre')

    if vaeske != 'Egendefinerte kjølevæskeegenskaper':
        c1,c2 = st.columns(2)
        with c1:
            st.metric("Prandlt-tall til kjølevæske", f"{round(Pr,1)}")
        with c2:
            st.metric("Ledningsevne til kjølevæske (k)", f"{round(ledningsevne,3)} W/mK")
        c1,c2 = st.columns(2)
        with c1:
            st.metric("Tetthet til kjølevæske", f"{round(tetthet)} kg/m\N{SUPERSCRIPT THREE}")
        with c2:
            st.metric("Viskositet til kjølevæske", f"{round(viskositet,5)} Pa s")

    c1,c2 = st.columns(2)
    with c1:
        st.metric("Nødvendig rørdiameter", f"{round(indre_diam_faktisk,1)} mm")
    with c2:
        st.metric("Nødvendig veggtykkelse", f"{round(tykk_faktisk,1)} mm")

    if ror_knapp == True:
        if tykk_tabell == tykk_tabell:
            c1,c2 = st.columns(2)
            with c1:
                st.metric("Rørdiameter fra tabell", f"{indre_diam_tabell} mm")
                st.metric("Vekt til rør", f"{rorvekt} kg/m")
            with c2:
                st.metric("Veggtykkelse fra tabell", f"{tykk_tabell} mm")
                
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Massestrøm", f"{round(massestrom,3)} kg/s")
    with c2:
        st.metric("Brønnlengde tur/retur", f"{lengde_bronn} m")


    c1, c2 = st.columns(2)
    with c1:
        st.metric("Friksjonsfaktor (f)", f"{round(frikfaktor,3)}")
    with c2:
        st.metric("Strømningshastighet", f"{round(hast,3)} m/s")

    st.markdown('---')
    st.subheader('Hovedresultater')
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Reynolds-tall (Re)", f"{round(Re,1)}")
    with c2:
        if trykkenhet == 'Pascal (Pa)':
            st.metric("Trykktap ($\Delta P$)", f"{round(trykktap,1)} Pa")
        elif trykkenhet == 'Megapascal (MPa)':
            st.metric("Trykktap ($\Delta P$)", f"{round(trykktap/10**6,3)} MPa")
        elif trykkenhet == 'Bar (bar)':
            st.metric("Trykktap ($\Delta P$)", f"{round(trykktap/10**5,3)} bar")
        elif trykkenhet == 'Pund per kvadrattomme (psi)':
            st.metric("Trykktap ($\Delta P$)", f"{round(trykktap/6894.75729,3)} psi")


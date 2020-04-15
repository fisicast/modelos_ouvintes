#importanto bibliotecas
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
from fbprophet import Prophet

#previsão diária
day = pd.read_csv("Fisicast_daily.csv", names = None) #dataframe de ouvintes por dia
day.rename(columns={"Time (UTC)": "ds", "Plays": "y"}, inplace = True) #renomeando colunas
day['ds'] = pd.to_datetime(day['ds'])

#separando dados de 2019 e 2020
day_2019 = day.loc[day['ds'].dt.year == 2019] 
day_2020 = day.loc[day['ds'].dt.year == 2020] 

#inicializando o Prophet e realizando ajuste
md = Prophet()
md.fit(day_2019)

#realizando previsão para os proximos 120 dias
future_d = md.make_future_dataframe(periods=120)
forecast_d = md.predict(future_d)

plt.figure(figsize = [12,8])

#grafico do numero de ouvintes
#ajuste utilizando o prophet e previsao do prophet
plt.plot(forecast_d['ds'], forecast_d['yhat'],'-',color = 'blue',label = 'Fit')
plt.fill_between(forecast_d['ds'], forecast_d['yhat_upper'],forecast_d['yhat_lower'], 
                 color = 'C0',alpha = 0.25, label = None)
plt.plot(day_2019['ds'], day_2019['y'], 'o', color = 'C1',label = '2019')
plt.plot(day_2020['ds'], day_2020['y'], 'o', color = 'C2',label = '2020')

plt.legend()
plt.xlabel('Date')
plt.ylabel('Listens')
plt.grid(True)
plt.xticks(rotation=30)


#trend
fig2 = md.plot_components(forecast_d)

for ax in fig2.axes:
    plt.setp(ax.get_xticklabels(), visible=True, rotation=30)


#analogamente previsao semanal
week = pd.read_csv("Fisicast_weekly.csv", names = None) #dataframe de ouvintes por semana
week.rename(columns={"Time (UTC)": "ds", "Plays": "y"}, inplace = True)
week['ds'] = pd.to_datetime(week['ds'])

week_2019 = week.loc[week['ds'].dt.year == 2019]
week_2020 = week.loc[week['ds'].dt.year == 2020]

mw = Prophet()
mw.fit(week_2019)

future_w = mw.make_future_dataframe(periods=120)
future_w.tail()

forecast_w = mw.predict(future_w)

plt.figure(figsize = [12,8])

#grafico do numero de ouvintes
#ajuste utilizando o prophet e previsao do prophet
plt.plot(forecast_w['ds'], forecast_w['yhat'],'-',color = 'blue',label = 'Fit')
plt.fill_between(forecast_w['ds'], forecast_w['yhat_upper'],forecast_w['yhat_lower'], 
                 color = 'C0',alpha = 0.25, label = None)
plt.plot(week_2019['ds'], week_2019['y'], 'o', color = 'C1',label = '2019')
plt.plot(week_2020['ds'], week_2020['y'], 'o', color = 'C2',label = '2020')

plt.legend()
plt.xlabel('Date')
plt.ylabel('Listens')
plt.grid(True)
plt.xticks(rotation=30)

fig2 = mw.plot_components(forecast_w)

for ax in fig2.axes:
    plt.setp(ax.get_xticklabels(), visible=True, rotation=30)




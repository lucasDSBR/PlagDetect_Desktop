#Importando bibliotecas
import re
import sys, os
import numpy as np
import plotly.offline
import PySimpleGUI as sg
import os, os.path #nova importação
import plotly.graph_objects as go #nova importação
import numpy as np #nova importação
from tika import parser #nova importção 
from nltk.tokenize import word_tokenize
from scipy.ndimage import gaussian_filter
from nltk.lm import MLE, WittenBellInterpolated
from nltk.util import ngrams, pad_sequence, everygrams

#Linha de código para escolher o estilo do Layout:
sg.theme('SystemDefaultForReal')


#contando o total de projetos na base de dados:
path, dirs, files = next(os.walk("projetos"))
file_count = len(files)
print(file_count)
def projSalvos(texto):
	texto
	valores_maximos = []
	valores_medios = []
	valores_arquivo = []
	textos_salvos_dados = []
	#gravar dados
	p = 0
	while p < file_count:
		print(p)
		textos_salvos_dados.insert(p, parser.from_file("projetos/"+files[p]))
		p = p + 1
	j = 0
	while j < file_count:
		print(j)
		#Variaveis:
		texto_fornecido = parser.from_file(texto)
		#Fornecendo dados para a variável "train_text" com o valor de pdf1_train para posteriormente ser analisado
		train_text = textos_salvos_dados[j]['content']

		print(train_text)
		# aplique o pré-processamento (remova o texto entre colchetes e chaves e rem punc)
		train_text = re.sub(r"\[.*\]|\{.*\}", "", train_text)
		train_text = re.sub(r'[^\w\s]', "", train_text)

		# definir o número ngram
		n = 5

		# preencher o texto e tokenizar
		training_data = list(pad_sequence(word_tokenize(train_text), n,
		                                  pad_left=True,
		                                  left_pad_symbol="<s>"))

		# gerar ngrams
		ngrams = list(everygrams(training_data, max_len=n))

		# build ngram language models
		model = WittenBellInterpolated(n)
		model.fit([ngrams], vocabulary_text=training_data)

		#Fornecendo dados para a variável "testt_text" com o valor de pdf2_test para posteriormente ser comparado com o arquivo de treinamento
		test_text = texto_fornecido['content']
		test_text = re.sub(r'[^\w\s]', "", test_text)

		# Tokenize e preencha o texto
		testing_data = list(pad_sequence(word_tokenize(test_text), n,
		                                 pad_left=True,
		                                 left_pad_symbol="<s>"))

		# atribuir pontuações
		scores = []
		for i, item in enumerate(testing_data[n-1:]):
		    s = model.score(item, testing_data[i:i+n-1])
		    scores.append(s)

		scores_np = np.array(scores)

		# definir largura e altura
		width = 8
		height = np.ceil(len(testing_data)/width).astype("int64")

		# copiar pontuações para matriz em branco retangular
		a = np.zeros(width*height)
		a[:len(scores_np)] = scores_np
		diff = len(a) - len(scores_np)

		# aplique suavização gaussiana para estética
		a = gaussian_filter(a, sigma=1.0)

		# remodelar para caber no retângulo
		a = a.reshape(-1, width)

		# rótulos de formato
		labels = [" ".join(testing_data[i:i+width]) for i in range(n-1, len(testing_data), width)]
		labels_individual = [x.split() for x in labels]
		labels_individual[-1] += [""]*diff
		labels = [f"{x:60.60}" for x in labels]

		# criar mapa de calor
		fig = go.Figure(data=go.Heatmap(
		                z=a, x0=0, dx=1,
		                y=labels, zmin=0, zmax=1,
		                customdata=labels_individual,
		                hovertemplate='%{customdata} <br><b>Score:%{z:.3f}<extra></extra>',
		                colorscale="burg"))
		fig.update_layout({"height":height*40, "width":1000, "font":{"family":"Courier New"}})
		
		#Armazenando dados dos scores para mostrar posteriormente
		valores_scores = np.array(scores)
		
		buscar_max = 0.9000000000000000 #Nivel alto de plágio

		buscar_med = 0.8000000000000000 #Nível acima da média
		

		maximo = np.where(valores_scores > buscar_max)[0]
		medio = np.where(valores_scores > buscar_med)[0]
		valores_maximos.insert(j, len(maximo))
		valores_medios.insert(j, len(medio))
		valores_arquivo.insert(j, files[j])
		#print('ok')
		#print(maximo)
		j = j + 1

	#buscando arquivo com maior nível igualdade:
	val_maximo = np.array(valores_maximos)
	val_medio = np.array(valores_medios)
	busc_val_max = 1050
	busc_val_med = 500
	maxx = np.where(val_maximo > busc_val_max)[0]
	medd = np.where(val_medio > busc_val_med)[0]
	if len(maxx) == 0:
		layout = [[sg.Text('RESULTADO DA ANALISE:')],
				  [sg.Text('Não foi encontrado nenhum arquivo que tenha similaridade com seu projeto.')],
				  [sg.Button('Sair', key='sair')]]
		window = sg.Window('Resultado', layout, icon=r'icon.ico')
		event, values = window.read()
	elif len(maxx) > 0:
		layout = [[sg.Text('RESULTADO DA ANALISE:')],
				  [sg.Text('O projeto encontrado que possui uma maior similaridade com o seu foi: ')],
				  [sg.Text('Projeto: '+files[int(maxx)])],
				  [sg.Text('Mais de 1000 anagramas com pontuações maiores que 0.90000...\n(Valor considerado alto em relação a similaridade.')],
				  [sg.Button('Sair', key='sair')]]
		window = sg.Window('Resultado', layout, icon=r'icon.ico')
		event, values = window.read()

#Função para analisar os arquivos separados:
def arquiSep(texto1, texto2):
	#variaveis em que os dados serão armazenados(localização dos arquivos):
	pdf1_train = parser.from_file(texto1)
	pdf2_test = parser.from_file(texto2)
	
	print(pdf1_train)
	print(pdf2_test)
	#Fornecendo dados para a variável "train_text" com o valor de pdf1_train para posteriormente ser analisado
	train_text = pdf1_train['content']


	# aplique o pré-processamento (remova o texto entre colchetes e chaves e rem punc)
	train_text = re.sub(r"\[.*\]|\{.*\}", "", train_text)
	train_text = re.sub(r'[^\w\s]', "", train_text)

	# definir o número ngram
	n = 5

	# preencher o texto e tokenizar
	training_data = list(pad_sequence(word_tokenize(train_text), n,
	                                  pad_left=True,
	                                  left_pad_symbol="<s>"))

	# gerar ngrams
	ngrams = list(everygrams(training_data, max_len=n))

	# build ngram language models
	model = WittenBellInterpolated(n)
	model.fit([ngrams], vocabulary_text=training_data)

	#Fornecendo dados para a variável "testt_text" com o valor de pdf2_test para posteriormente ser comparado com o arquivo de treinamento
	test_text = pdf2_test['content']

	test_text = re.sub(r'[^\w\s]', "", test_text)

	# Tokenize e preencha o texto
	testing_data = list(pad_sequence(word_tokenize(test_text), n,
	                                 pad_left=True,
	                                 left_pad_symbol="<s>"))

	# atribuir pontuações
	scores = []
	for i, item in enumerate(testing_data[n-1:]):
	    s = model.score(item, testing_data[i:i+n-1])
	    scores.append(s)

	scores_np = np.array(scores)

	# definir largura e altura
	width = 8
	height = np.ceil(len(testing_data)/width).astype("int64")

	# copiar pontuações para matriz em branco retangular
	a = np.zeros(width*height)
	a[:len(scores_np)] = scores_np
	diff = len(a) - len(scores_np)

	# aplique suavização gaussiana para estética
	a = gaussian_filter(a, sigma=1.0)

	# remodelar para caber no retângulo
	a = a.reshape(-1, width)

	# rótulos de formato
	labels = [" ".join(testing_data[i:i+width]) for i in range(n-1, len(testing_data), width)]
	labels_individual = [x.split() for x in labels]
	labels_individual[-1] += [""]*diff
	labels = [f"{x:60.60}" for x in labels]

	# criar mapa de calor
	fig = go.Figure(data=go.Heatmap(
	                z=a, x0=0, dx=1,
	                y=labels, zmin=0, zmax=1,
	                customdata=labels_individual,
	                hovertemplate='%{customdata} <br><b>Score:%{z:.3f}<extra></extra>',
	                colorscale="burg"))
	fig.update_layout({"height":height*40, "width":1000, "font":{"family":"Courier New"}})
	#Criando arquivo em .HTML com os dados:
	plotly.offline.plot(fig, filename='resultado.html', auto_open=False)
	#Mostrando notiicação no canto inferior direito da tela com uma mensagem:
	sg.SystemTray.notify('PlagDetect:', 'Arquivos analisados com sucesso!. Abra o arquivo "resultado.html" para ver o resultado.')
	#Mostrando um popup com os dados dos arquivos analisados:
	sg.popup('Informações','Número de ngrams: '+str(len(ngrams)), '\nComprimento dos dados de teste: '+str(len(testing_data)), '\nLargura/altura: '+str(width)+','+str(height), icon=r'icon.ico')
#Armazenando dados dos scores para mostrar posteriormente
	valores_scores = np.array(scores)
		
	buscar_max = 0.9000000000000000 #Nivel alto de plágio

	buscar_med = 0.8000000000000000 #Nível acima da média
		
	maximo = np.where(valores_scores > buscar_max)[0]
	medio = np.where(valores_scores > buscar_med)[0]
	valores_maximos.insert(j, len(maximo))
	valores_medios.insert(j, len(medio))
	valores_arquivo.insert(j, files[j])
	print('ok')
	print(medio)

	#buscando arquivo com maior nível igualdade:
	val_maximo = np.array(valores_maximos)
	val_medio = np.array(valores_medios)
	busc_val_max = 1090
	busc_val_med = 500
	maxx = np.where(val_maximo > busc_val_max)[0]
	medd = np.where(val_medio > busc_val_med)[0]
	print(val_maximo)
	print(len(maxx))
############################## Layout Principal PARTE FRONT-END: ################################


#Layout que será mostrado para o usuário
layout = [[sg.Image(r'logo.png')],
          [sg.Button('Analisar projeto com a base de dados', key='op11', size=(28, 1))],
          [sg.Button('Analisar projetos separados', key='op22', size=(28, 1))],
          [sg.Button('Treinar o PlagDetect', key='op33', size=(28, 1))],
          [sg.Button('Sair', key='sair', size=(28, 1))]]
window = sg.Window('PlagDetect', layout, icon=r'icon.ico', element_justification='c', size=(800, 400))

event, values = window.read()


#BACK-END do layout e etc:
while True:
    event, values = window.read()
    #Se o usuário clicar no botão com a 'key' op11 será mostrado uma aba para analisar todos os grupo(ainda tem que ser desenvolvido):
    if event == 'op11':
    	#Layout que será mostrado:
        layout = [[sg.Text('Total de projetos na base de dados:')],
        		  [sg.Text('Selecione o arquivo que deseja analisar:')],
        		  [sg.Input(), sg.FileBrowse()],
                  [sg.B('Sair', key='sair'), sg.B('Iniciar', key='ini')]]
        window = sg.Window('PlagDetect', layout, icon=r'icon.ico')
        event, values = window.read()
        #Condicionais do layout:
        while True:
            if event == 'ini':
            	projSalvos(values[0])
            	break
            elif event == 'sair':
            	break
    #Se o usuário clicar no botão com a 'key' op22 será mostrado uma aba para analisar trabalhos em pastas separadas:
    elif event == 'op22':
    	#Layout que será mostrado:
        layout = [[sg.Text('Selecione o primeiro texto:')],
        		  [sg.Input(), sg.FileBrowse()],
        		  [sg.Text('Selecione o segundo texto:')],
        		  [sg.Input(), sg.FileBrowse()],
                  [sg.Button('Sair', key='sair'), sg.Button('Iniciar', key='ini')]]
        window = sg.Window('PlagDetect', layout, icon=r'icon.ico')
        event, values = window.read()
        #Condicionais:
        while True:
            if event == 'ini':
            	#Se o usuário clicar em 'INICIAR', o sistema chamará a função 'arquiSep' que é usada para analisar os arquivos separados
            	#juntamente a isso, mandará o diretório dos arquivos selecionados(values[0] e values[1]).
                arquiSep(values[0], values[1])
                break
            elif event in (sg.WINDOW_CLOSED, 'sair'):
                break
    elif event == 'op33':
    	layout = [[sg.Text('#')],
    			  [sg.Text('AINDA ESTÁ EM DESENVOLVIMENTO')],
    			  [sg.Button('Sair', key='sair'), sg.Button('Gravar', key='grav')]]
    	window = sg.Window('PlagDetect', layout, icon=r'icon.ico')
    	event, values = window.read()
    	while True:
    		if event == 'grav':
    			gravProj(values[0])
    			break
    		elif event == 'sair':
    			break
    	#layout para enviar arquivos para treinamento:
    elif event in (sg.WINDOW_CLOSED, 'sair'):
        break
window.close()
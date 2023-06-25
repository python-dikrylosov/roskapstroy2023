import pandas as pd
import time
import nltk
nltk.download('punkt') # для токенизации необходимо скачать модуль
import pymorphy2 # библиотека для лемматизации
import re
from nltk.corpus import stopwords
filename = "Датасет.xlsx"
data = pd.read_excel(filename)
#print(data)
#print(data.shape)
sdfgdfg = ["Арматурщик","Инженер","Слесарь-Ремонтник"]
for i in range(data.shape[0]):

        import requests
        import bs4

        cookies = {
                '_ym_uid': '1616661207763271544',
                '_ym_d': '1663424427',
                '__ddg1_': '83Ffnluv63rwe2MHhPNO',
                'tmr_lvid': 'ec909d6bd0626bcb7353ff87094f443c',
                'tmr_lvidTS': '1616661206317',
                '_ga': 'GA1.2.1283363942.1666370669',
                'tmr_detect': '1%7C1667230005197',
                '_gid': 'GA1.2.1931147048.1667230005',
                '_ym_isad': '1',
                'tmr_reqNum': '337',
        }

        headers = {
                'authority': 'pln-pskov.ru',
                'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
                'accept-language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
                # Requests sorts cookies= alphabetically
                # 'cookie': '_ym_uid=1616661207763271544; _ym_d=1663424427; __ddg1_=83Ffnluv63rwe2MHhPNO; tmr_lvid=ec909d6bd0626bcb7353ff87094f443c; tmr_lvidTS=1616661206317; _ga=GA1.2.1283363942.1666370669; tmr_detect=1%7C1667230005197; _gid=GA1.2.1931147048.1667230005; _ym_isad=1; tmr_reqNum=337',
                'referer': 'https://pln-pskov.ru/',
                'sec-ch-ua': '"Chromium";v="106", "Google Chrome";v="106", "Not;A=Brand";v="99"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"macOS"',
                'sec-fetch-dest': 'document',
                'sec-fetch-mode': 'navigate',
                'sec-fetch-site': 'same-origin',
                'sec-fetch-user': '?1',
                'upgrade-insecure-requests': '1',
                'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36',
        }

        data_link_resource = data["link_resource"][i]
        print(data_link_resource)
        response = requests.get(data_link_resource,cookies=cookies, headers=headers)
        print(response)
        news = bs4.BeautifulSoup(response.text)
        print(news)
        links = news.find_all({'class': "g-user-content"},{'data-qa':'vacancy-description'})
        print(links)
        for i in range(4):
                time.sleep(1)
                print(i)


        print("\n\n",i,data["id"][i],"\n",          data["name(название)"][i],"\n",data["responsibilities(Должностные обязанности)"][i],"\n",)

        ru_text = str(data["responsibilities(Должностные обязанности)"][i])
        data_text = ru_text.lower()
        print()
        print("Получаем текст из строки Должностные Обязанности ")
        #Мешок слов
        # для мешка слов и TF-IDF импортируем:
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        import nltk
        import pymorphy2
        import re
        import re
        from nltk.corpus import stopwords
        nltk.download('punkt')  # для токенизации необходимо скачать модуль
        nltk.download('stopwords')
        import pymorphy2  # библиотека для лемматизации
        from sklearn.feature_extraction.text import CountVectorizer
        import textdistance
        import nltk
        import pymorphy2
        sen = nltk.sent_tokenize(ru_text)
        print("Токенизируем")
        print(sen)

        morph = pymorphy2.MorphAnalyzer()
        tokenized = []
        for i in sen:
                tok_sen = ''
                txt = re.findall(r'[а-я]+', i.lower())
                for j in txt:
                        if j not in stopwords.words('russian'):
                                w = morph.parse(j)[0].normal_form
                                if tok_sen == '':
                                        tok_sen += w
                                else:
                                        tok_sen += (' ' + w)
                tokenized.append(tok_sen)
        print("Токенизируем")
        print(tokenized)

        print("Приводим весь текст в нижний регистр")
        txt = re.findall(r'[а-я]+', sen[0].lower())
        print(txt)


        vectorizer = CountVectorizer()
        bag = vectorizer.fit_transform(tokenized)
        print("сконвертировать набор текстов в матрицу токенов")
        print(vectorizer.vocabulary_)
        print("подсчитываем")
        print(bag.toarray())
        print(" \n")
        print(bag)

        morph = pymorphy2.MorphAnalyzer()

        from string import punctuation
        import gensim.downloader as api
        model = api.load("word2vec-ruscorpora-300")

        print('Проверим косинусное сходство на наличие условия в тексте')
        text1 = ru_text
        ru_text2 = str("условия")
        text2 = str(ru_text2)

        def lemitiz(text: str):
                for elem in punctuation:
                        text = text.replace(elem, '')
                text = text.lower().split()

                T = []
                for word in text:
                        if word in nltk.corpus.stopwords.words('russian'):
                                continue
                        T.append(morph.parse(word)[0].normal_form)
                return T


        T1 = lemitiz(text1)
        T2 = lemitiz(text2)

        cos = textdistance.cosine(T1, T2)
        print(f'Тексты схожи на {100 * cos}%.')
        if (100 * cos) > 0:
                print("Условия есть" )
                time.sleep(10)
        elif (100 * cos) == 0:
                print("Условий не обнаружено")
        time.sleep(2)






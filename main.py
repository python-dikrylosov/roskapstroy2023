import pandas as pd
import time
import nltk
nltk.download('punkt') # для токенизации необходимо скачать модуль
import pymorphy2 # библиотека для лемматизации
import re
from nltk.corpus import stopwords
filename = "Датасет.xlsx"
data = pd.read_excel(filename)
print(data)
print(data.shape)
sdfgdfg = ["Арматурщик","Инженер","Слесарь-Ремонтник"]
for i in range(data.shape[0]):
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
                print("Условия есть")
                time.sleep(10)
        elif (100 * cos) == 0:
                print("Условий не обнаружено")
        time.sleep(2)






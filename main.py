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

        #Мешок слов
        # для мешка слов и TF-IDF импортируем:
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        import nltk
        nltk.download('punkt')  # для токенизации необходимо скачать модуль
        import pymorphy2  # библиотека для лемматизации
        nltk.download('punkt')
        sen = nltk.sent_tokenize(ru_text)
        print(sen)
        nltk.download('stopwords')
        import pymorphy2
        import re
        from nltk.corpus import stopwords

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

        print(tokenized)
        print(sen)
        print(sen[0].lower)
        import re
        txt = re.findall(r'[а-я]+', sen[0].lower())
        print(txt)

        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer()
        bag = vectorizer.fit_transform(tokenized)
        print(vectorizer.vocabulary_)
        print(bag.toarray())
        print(bag)
        import textdistance
        import nltk

        nltk.download('stopwords')
        import pymorphy2

        morph = pymorphy2.MorphAnalyzer()
        from string import punctuation
        import gensim.downloader as api
        model = api.load("word2vec-ruscorpora-300")

        print('''Проверим косинусное сходство двух рускоязычных текстов.''')
        text1 = ru_text
        ru_text2 = str(data["responsibilities(Должностные обязанности)"][0])
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
        time.sleep(10)






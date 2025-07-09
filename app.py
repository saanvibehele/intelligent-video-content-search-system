from flask import Flask, request, redirect, url_for, render_template
import os
import multiprocessing as mp
from process_video import save_video, process_each_frame, detect_n_save
import pickle
import pandas as pd
import spacy, string, re
from spacy.lang.en.stop_words import STOP_WORDS
import numpy as np
import gensim
from gensim import corpora
from operator import itemgetter
import cv2
import time


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/videos'


def get_frame(search_query):
    with open("data2.pkl", "rb") as file:
        data = pickle.load(file)

    flattened_data = []

    for frame_no, frame_data in data.items():
        desc = frame_data.get('desc', '')
        objects = [obj for obj in frame_data.keys() if obj != 'desc']
        if not objects:
            objects = ['none']
        for obj in objects:
            if obj != 'none':
                desc += " "
                desc += obj
        flattened_data.append({'frame_no': frame_no, 'desc': desc, 'object': objects})
    
    df = pd.DataFrame(flattened_data)
    df.to_csv("frame_data1.csv")
    df = pd.read_csv("frame_data1.csv")
    df1 = df.drop(columns=['Unnamed: 0', 'object'])
    #create list of punctuations and stopwords
    spacy_nlp = spacy.load('en_core_web_sm')
    punctuations = string.punctuation
    stop_words = spacy.lang.en.stop_words.STOP_WORDS

    #function for data cleaning and processing
    #This can be further enhanced by adding / removing reg-exps as desired.

    def spacy_tokenizer(sentence):
        sentence = re.sub(r'[^\w\s]',' ',sentence)
        tokens = spacy_nlp(sentence)
        tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]
        tokens = [word for word in tokens if word not in stop_words and word not in punctuations and len(word) > 2]
        return tokens
    
    print ('Cleaning and Tokenizing...')
    df1['desc_token'] = df1['desc'].map(lambda x: spacy_tokenizer(x))
    frame_data = df1['desc_token']
    dictionary = corpora.Dictionary(frame_data)
    #list of few which which can be further removed
    stoplist = set('hello and if this can would should could tell ask stop come go')
    stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
    dictionary.filter_tokens(stop_ids)

    corpus = [dictionary.doc2bow(desc) for desc in frame_data]
    word_frequencies = [[(dictionary[id], frequency) for id, frequency in line] for line in corpus[0:3]]

    data_tfidf_model = gensim.models.TfidfModel(corpus, id2word=dictionary)
    data_lsi_model = gensim.models.LsiModel(data_tfidf_model[corpus], id2word=dictionary, num_topics=300)

    #Serialize and Store the corpus locally for easy retrival whenver required.
    gensim.corpora.MmCorpus.serialize('data_tfidf_model_mm', data_tfidf_model[corpus])
    gensim.corpora.MmCorpus.serialize('data_lsi_model_mm',data_lsi_model[data_tfidf_model[corpus]])

    #Load the indexed corpus
    data_tfidf_corpus = gensim.corpora.MmCorpus('data_tfidf_model_mm')
    data_lsi_corpus = gensim.corpora.MmCorpus('data_lsi_model_mm')

    #Load the MatrixSimilarity
    from gensim.similarities import MatrixSimilarity
    frame_index = MatrixSimilarity(data_lsi_corpus, num_features = data_lsi_corpus.num_terms)

    def search_in_frames(search_term):
        query_bow = dictionary.doc2bow(spacy_tokenizer(search_term))
        query_tfidf = data_tfidf_model[query_bow]
        query_lsi = data_lsi_model[query_tfidf]

        frame_index.num_best = 5

        frame_list = frame_index[query_lsi]

        # frame_list.sort(key=itemgetter(1), reverse=True)
        frame_names = []

        for j, frame in enumerate(frame_list):

            frame_names.append (
                {
                    'relevance': round((frame[1] * 100),2),
                    'frame_no': df1['frame_no'][frame[0]],
                    'desc': df1['desc'][frame[0]]
                }

            )
            if j == (frame_index.num_best-1):
                break

        return pd.DataFrame(frame_names, columns=['relevance','frame_no','desc'])

    df_r = search_in_frames(search_query)
    result_frame = df_r['frame_no'][0]
    match = re.search(r'\d+', result_frame)
    time_in_seconds = int(match.group())
    print("time in seconds is: ",time_in_seconds)

    video_no = int(time_in_seconds/32) + 1
    time_in_seconds = time_in_seconds % 32
    video_path = f'output_{video_no}.webm' 
    print("video path: ",video_path)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(fps * time_in_seconds)
    print("frame_ number: ", frame_number)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    fourcc = cv2.VideoWriter_fourcc(*'vp80')
    videop = 'show_output.webm'
    out_path = os.path.join(app.config['UPLOAD_FOLDER'], videop)
    out = cv2.VideoWriter(out_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        
        if ret:
            # cv2.imshow('Video', frame)
            out.write(frame)
            
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    return videop




def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov', 'mkv', 'webm'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        q = mp.Queue()
        data = {}
        p1 = mp.Process(target=save_video, args=(filepath, q,))
        p2 = mp.Process(target=process_each_frame, args=(q, data,))
        p1.start()
        time.sleep(0.5)
        p2.start()
        p1.join()
        p2.join()
        print("filename in upload: ",filename)
        print(type(filename))
        return render_template('text_input.html', message="", filename=filename)
    return redirect(request.url)


@app.route('/submit_text', methods=['POST'])
def submit_text():
    user_text = request.form['text_input']
    filename = request.form['filename']
    # Handle the text input as needed (e.g., save it to a file or database)
    # with open('user_text.txt', 'a') as f:
    #     f.write(f"Filename: {filename}, User Text: {user_text}\n")
    message = f'Text received: {user_text} for file: {filename}. Please enter more text or review the video again.'
    filename = get_frame(user_text)
    print("file name: ", filename)
    print(type(filename))
    return render_template('text_input.html', message="", filename=filename)

if __name__ == '__main__':
    app.run(debug=True)

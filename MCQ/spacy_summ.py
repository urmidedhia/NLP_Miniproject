# NLP Pkgs
import spacy 
nlp = spacy.load("en_core_web_sm")
# Pkgs for Normalizing Text
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
# Import Heapq for Finding the Top N Sentences
from heapq import nlargest

from nltk.tokenize import sent_tokenize


text = '''India is the undisputed home of the majestic tiger, harboring over 70% of the world's entire population. These striped wonders, also known locally as 'baagh,' 'puli,' or 'sher,' hold the prestigious title of India's National Animal since 1972. Bengal tigers, a subspecies known for their impressive size, reign supreme across the country's diverse landscapes. From the tall grasslands to the mysterious mangrove swamps, these apex predators thrive in tropical and sub-tropical forests, climbing to elevations over 6,000 feet in the mountains. Their adaptability is a marvel, as they stalk prey through dense shola forests, a unique ecosystem of stunted trees draped in moss. Sadly, their magnificence is marked by the constant threat of endangerment.

Project Tiger, a national conservation effort launched in 1973, has been instrumental in reversing the decline of their numbers. Through a network of tiger reserves, these protected areas provide safe havens for tigers to roam and breed. Camera traps and meticulous monitoring efforts by the National Tiger Conservation Authority paint a cautiously optimistic picture. From a worrying low in 2010, tiger populations have shown a significant increase, thanks to relentless anti-poaching efforts and community outreach programs.

However, challenges remain. Habitat loss due to human encroachment and poaching for their body parts continue to threaten these magnificent creatures. Yet, India's commitment to tiger conservation serves as a beacon of hope.  The sight of a tiger, its stripes flashing through the foliage, is a powerful reminder of the beauty and delicate balance of our natural world.  Efforts to conserve them go beyond protecting a species; they are vital for safeguarding the intricate web of life that sustains us all.'''


def spacy_text_summarizer(raw_docx):
    raw_text = raw_docx
    no_sents = len(sent_tokenize(raw_text))
    docx = nlp(raw_text)
    stopwords = list(STOP_WORDS)
    # Build Word Frequency # word.text is tokenization in spacy
    word_frequencies = {}  
    for word in docx:  
        if word.text not in stopwords:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1


    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
    # Sentence Tokens
    sentence_list = [ sentence for sentence in docx.sents ]

    # Sentence Scores
    sentence_scores = {}  
    for sent in sentence_list:  
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if len(sent.text.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.text.lower()]


    summarized_sentences = nlargest(round(no_sents*0.4), sentence_scores, key=sentence_scores.get)
    final_sentences = [ w.text for w in summarized_sentences ]
    summary = ' '.join(final_sentences)
    return summary
    






# def summarizerTool(pdf_path):
#     model = Summarizer()
#     result = model(text, min_length=60, max_length = 500 , ratio = 0.5)

#     summarized_text = ''.join(result)
#     return summarized_text

# summary2 = summarizerTool(text)
# print('LIBRARY SUMMARY: \n', summary2)
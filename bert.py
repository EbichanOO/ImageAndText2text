from bertChainer import modeling

x = input()
in_words = [int.from_bytes(j.encode('utf-8'), 'big') for j in x]
#utf-8 is under int 14925759
BERT = modeling.BertModel(14925759)
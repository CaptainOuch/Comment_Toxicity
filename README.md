# Определение лица и насколько оно секси

## Данные
Данные были взяты с соревнование на kaggle
<p align="center">
<img width=500 src= "https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/overview"/>
</p>


## Обучение модели

Основная особенность модели в двустороннем lstm слое. Это позволяет модели наблюдать корреляции слов не только слева-направо, но и справа-налево.

```python
model = Sequential()
# Create the embedding layer 
model.add(Embedding(MAX_FEATURES+1, 32))
# Bidirectional LSTM Layer
model.add(Bidirectional(LSTM(32, activation='tanh')))
# Feature extractor Fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
# Final layer 
model.add(Dense(6, activation='sigmoid'))
```

<p align="center">
<img width=500 src= "https://user-images.githubusercontent.com/38643187/244715459-625d7a1d-05b2-429a-bb34-bbf0cf73fd54.png"/>
</p>

## Оценка модели

```python
print(f'Precision: {pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')
```

    Precision: 0.9288560152053833, Recall:0.9220339059829712, Accuracy:0.5115346312522888
    

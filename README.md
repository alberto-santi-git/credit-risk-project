# Credit Risk Prediction

Modello di classificazione per stimare la probabilità di default su prestiti al consumo, a partire dal [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) (32.581 richieste di prestito).

## Il problema e il design della soluzione

L'obiettivo è predire `loan_status` (0 = rimborsato, 1 = default) a partire dalle caratteristiche del richiedente e del prestito.

Due colonne del dataset — `loan_grade` e `loan_int_rate` — sono assegnate dalla banca **dopo** una propria valutazione del rischio del cliente. Includerle nel training rischia di far "copiare" al modello un giudizio già espresso da altri, invece di impararlo dai dati grezzi del cliente. Per questo il progetto confronta due scenari:

- **`full`**: tutte le feature disponibili, incluse `loan_grade`/`loan_int_rate`
- **`clean`**: solo le feature note *prima* della valutazione del rischio — il vero esercizio di credit scoring, ed è il modello che viene salvato e messo in produzione nell'app demo

## Pipeline

1. **Data cleaning**: rimozione di outlier evidenti (`person_age` e `person_emp_length` con valori chiaramente errati, es. età >100 anni), imputazione dei missing su `person_emp_length` e `loan_int_rate` con la mediana
2. **Feature engineering**: rapporto storico creditizio/età, reddito residuo dopo il prestito, reddito per anno di esperienza lavorativa
3. **Modellazione**: confronto tra Logistic Regression (baseline interpretabile), Random Forest ed XGBoost, ciascuno tunato con `GridSearchCV` e cross-validation stratificata a 5 fold, ottimizzando ROC-AUC
4. **Valutazione**: ROC-AUC e PR-AUC su test set stratificato (l'accuracy da sola è poco informativa su un target sbilanciato ~22% default), confusion matrix, feature importance
5. **Deploy**: app Streamlit che carica il modello `clean` salvato e restituisce una probabilità di default in tempo reale

## Struttura del repo

```
├── notebooks/
│   └── credit_risk_analysis.ipynb   # EDA, cleaning, training, valutazione, confronto full vs clean
├── src/
│   ├── data_prep.py                  # cleaning e feature engineering (riusato da notebook e app)
│   └── modeling.py                   # preprocessing, training, valutazione dei modelli
├── models/                           # modello finale salvato (generato dal notebook)
├── data/                             # posiziona qui credit_risk_dataset.csv (non incluso nel repo)
├── app.py                            # demo Streamlit
└── requirements.txt
```

## Come eseguirlo

```bash
pip install -r requirements.txt

# 1. Scarica il dataset da Kaggle e posizionalo in data/credit_risk_dataset.csv

# 2. Esegui il notebook (genera models/best_credit_risk_model.pkl)
jupyter notebook notebooks/credit_risk_analysis.ipynb

# 3. Avvia la demo
streamlit run app.py
```

## Possibili estensioni

- Calibrazione delle probabilità (Platt scaling / isotonic regression) per un uso diretto come PD in un contesto IFRS9
- Analisi di fairness su variabili come `person_home_ownership`
- Soglia di decisione ottimizzata sul costo asimmetrico falso negativo vs falso positivo, invece della soglia di default a 0.5

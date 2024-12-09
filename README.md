Projeto para classificar dois tipos de uvas passas (`Kecimen` e `Besni`) usando um modelo de rede neural MLP.

---

## **Como Executar**

1. **Pré-processar os Dados**  
   Execute o script para preparar os dados:
   ```bash
   python scripts/data_preprocessing.py
   ```

2. **Treinar o Modelo**  
   Execute o script para treinar o modelo:
   ```bash
   python scripts/model_training.py
   ```

3. **Avaliar o Modelo**  
   Execute o script para gerar gráficos e relatórios:
   ```bash
   python scripts/evaluate_model.py
   ```

---

## **Arquivos Importantes**

- `data/`: Contém o dataset original e os dados processados.
- `models/mlp_model.h5`: Modelo treinado.
- `results/`: Gráficos e resultados da avaliação.
- `scripts/`: Scripts do projeto.

---

## **Dependências**

  ```bash
  pip install pandas numpy matplotlib tensorflow scikit-learn
  ```
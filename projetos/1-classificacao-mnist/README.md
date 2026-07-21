# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo: Carlos Eduardo Batista Diniz**

### 1️⃣ Resumo da Arquitetura do Modelo

A CNN implementada em `train_model.py` é composta por 3 blocos convolucionais sequenciais, cada um seguido de Batch Normalization e Max Pooling:

- **Bloco 1:** Conv2D (32 filtros, 3x3) → BatchNormalization → MaxPooling2D (2x2)
- **Bloco 2:** Conv2D (64 filtros, 3x3) → BatchNormalization → MaxPooling2D (2x2)
- **Bloco 3:** Conv2D (128 filtros, 3x3) → BatchNormalization → MaxPooling2D (2x2)

Após os blocos convolucionais, a saída é achatada (`Flatten`) e passa por uma camada de `Dropout(0.5)` para regularização, evitando overfitting, antes da camada de saída `Dense(10, activation="softmax")`, responsável pela classificação entre as 10 classes de dígitos (0-9).

O modelo possui um total de **105.098 parâmetros** (~410 KB), sendo 104.650 treináveis e 448 não-treináveis (referentes às camadas de Batch Normalization).

Para o treinamento, foi utilizado `validation_split=0.1` (10% dos dados de treino reservados para validação) e a técnica de **EarlyStopping**, monitorando a métrica `val_loss` com `patience=3` e `restore_best_weights=True`, garantindo que o modelo final salvo seja o de melhor desempenho em validação, evitando treinar além do necessário. O treinamento foi configurado para até 15 épocas, mas foi interrompido automaticamente pelo EarlyStopping na 7ª época.

### 2️⃣ Bibliotecas Utilizadas

- **TensorFlow** 2.21.0
- **Keras** 3.12.3
- **NumPy** 2.2.6

### 3️⃣ Técnica de Otimização do Modelo

Em `optimize_model.py`, o modelo treinado (`model.h5`) foi convertido para o formato TensorFlow Lite (`model.tflite`) utilizando o `tf.lite.TFLiteConverter`. A técnica de otimização aplicada foi a **Dynamic Range Quantization**, configurada através de `converter.optimizations = [tf.lite.Optimize.DEFAULT]`. Essa técnica reduz a precisão numérica dos pesos do modelo (de ponto flutuante de 32 bits para representações de menor precisão), diminuindo significativamente o tamanho do arquivo e o custo computacional da inferência, com impacto mínimo na acurácia — sendo especialmente adequada para implantação em dispositivos de borda (Edge AI) com recursos limitados.

### 4️⃣ Resultados Obtidos

| Métrica | Valor |
|---|---|
| Acurácia de validação final | **98,97%** |
| Acurácia no conjunto de teste | **99,12%** |
| Tamanho do `model.h5` | **1.290,91 KB** (~1,26 MB) |
| Tamanho do `model.tflite` | **113,96 KB** |
| Redução de tamanho após otimização | **91,2%** |

### 5️⃣ Comentários Adicionais (Opcional)

Um ponto que vale mencionar é a relação entre o tamanho do modelo e o desempenho: a quantização dinâmica reduziu o `model.tflite` para cerca de 9% do tamanho do `model.h5` original, sem impacto perceptível na acurácia (validação e teste ficaram praticamente no mesmo patamar de antes da conversão). Isso mostra que, para um problema relativamente simples como o MNIST, boa parte da precisão de 32 bits dos pesos originais é redundante — o modelo consegue manter a mesma capacidade de decisão com bem menos informação numérica, o que é justamente o tipo de ganho que interessa para Edge AI, onde memória e poder de processamento são limitados.

Na arquitetura, optei por manter os blocos convolucionais relativamente enxutos (32-64-128 filtros) em vez de ir mais fundo, já que o MNIST não exige tanta capacidade representacional — uma rede maior tenderia só a aumentar o tempo de treino e o risco de overfitting sem ganho real de acurácia. O Dropout de 0.5 antes da camada final ajudou a manter a validação estável entre as épocas, e o EarlyStopping evitou treinar além do ponto em que o modelo já tinha parado de melhorar.

### 6️⃣ Exemplo de Inferência

Saída do terminal ao rodar `run_inference.py`:
```
Rodando inferencia em 5 amostras usando model.tflite:

Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4
```

As 5 amostras foram classificadas corretamente. Vale destacar que o modelo diferenciou bem o dígito 1 do 7, que são traços visualmente próximos e costumam gerar confusão em modelos menos robustos. Não houve degradação perceptível de desempenho após a quantização, o que confirma que a técnica de Dynamic Range Quantization manteve a capacidade de generalização do modelo mesmo com a redução de precisão dos pesos.


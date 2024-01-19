# Introduction

- RNN, LSTM 등의 순환 신경망 모델과 encoder-decoder 구조는 시계열, 언어 모델, 기계 번역 문제를 해결하는 방법으로 지속해서 발전해옴.

- 최근 연구에서는 factorization 기법과 conditional computation으로 연산 부분에서 효율적인 성능을 이루었지만, Recurrent 모델에서는 병렬화, sequence 데이터의 길이, 메모리 등 순차 연산에 제약이 남아있음.

- Attention 메커니즘은 sequence modeling과 변환 모델에서 필수적인 요소가 됨.

- Transformer은 attention 메커니즘을 전적으로 의존하여 훨씬 많은 병렬화를 허용하여 번역 품질에 있어서 state-of-art를 달성

# Background

- 순차 연산을 줄이기 위한 방법으로 ConvS2S, ByteNet 등의 모델은 CNN에 block을 쌓고, 병렬로 계산함. 
    
    하지만, 서로의 위치가 멀리 있는 데이터에 관해서 ConvS2S는 선형적으로, ByteNet은 대수적으로 연산하기 때문에 먼 거리의 의존성을 학습하기 어려워짐. 

    Transformer에서는 Multi-Head Attention을 사용하여 연산량을 줄임.

- Self-attention(infra-attention)은 하나의 sequence에서 다른 위치에 있는 sequence의 표현(represention)를 계산하기 위한 attention 메커니즘

    독해, 요약, 독립적인 문장 표현 등 다양한 작업에서 사용되고 있음.

- End-to-end memory network는 recurrent attention 메커니즘을 기반으로 하고, 간단한 언어 질문에 대한 대답과 언어 모델링 작업에서 좋은 성능을 보임.

- 결국 Transformer는 sequence-aligned RNN 또는 Convolution 사용 없이 전적으로 self-attention에 의존하여 입력과 출력의 표현(representation)을 계산

# Model Architecture

- 대부분의 sequence 변환 모델은 encoder-decoder 구조
 
    Encoder는 input sequence **x**(x1 ... xn)를 **z** (z1 ... zn)로 매핑

    Decoder는 주어진 **z**를 output sequence **y**(y1 ... ym)를 생성

    각 단계마다 이전에 생성된 sequence를 다음에 생성할 sequence의 입력으로 사용

- Transformer의 전체 아키텍처는 encoder와 decoder 모두 self-attention과 point-wise, fully-connected layer로 구성

    ## Encoder

    - 6개의 계층으로 구성

    - 각 계층은 두 개의 sub-layer로 구성
        - multi-head self-attention 메커니즘
        - position-wise fully connected feed-forward network
    
    - 각 sub-layer에 residual-connection 과 Layer normalization 적용

    ## Decoder

    - Encoder와 마찬가지로 6개의 계층으로 구성

    - 각 계층은 세 개의 sub-layer로 구성
        - masked multi-head self-attention 메커니즘
        + Encoder의 출력으로 multi-head attention 수행하는 layer
        - position-wise fully connected feed-forward network
    
    - 각 sub-layer에 residual-connection 과 Layer normalization 적용

    ## Attention

    - Query, key-value 쌍 벡터를 출력 벡터에 매핑

    ### Scaled Dot-Product Attention

    

    ### Multi-Head Attention

    ## Position-wise Feed-Forword Networks

    ## Embeddings and Softmax

    ## Positional Encoding

# 1. Attention Idea

- Decoder에서 출력 단어를 예측하는 시점(Time step)마다 Encoder에서의 전체 Sequence에서 해당 시점에 예측해야할 단어와 연관이 있는 입력 Sequence 부분에 조금 더 집중(attention)

# 2. Attention Function

- Attention(Q, K, V) = Attention Value
    
    - 주어진 Query에 대해서 모든 Key와의 유사도를 구함.
    - 유사도를 Key와 매핑되어 있는 Value에 반영
    - Attention Value는 유사도가 반영된 Value를 모두 더해서 리턴한 것


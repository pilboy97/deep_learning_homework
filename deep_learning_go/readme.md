# GO로 구현을 시도해 본 신경망 프로젝트(실패함)
이 프로젝트는 다음의 구조로 되어 있습니다.

- cmd : 메인 프로젝트를 저장하고 있습니다.
  - xor : xor을 학습해보려 시도해보았습니다만, 어째선지 제대로 되지 않았습니다.
- internal : 내부 패키지를 저장하고 있습니다.
  - mat : 행렬을 정의하고 있습니다.
    - v1 : 일반적인 연산을 지원하고 있습니다.
      - mat_test.go : 테스트 코드입니다.
      - mat.go : 행렬을 정의하고 있습니다.
    - v2 : 병렬처리를 도입해, 성능을 향상해보려 했습니다.
      - mat_test.go : 테스트 코드입니다.
      - mat.go : 행렬을 정의하고 있습니다.
  - nn : 신경망에 필요한 요소를 정의하고 있습니다.
    - adam.go : adam optimizer입니다.
    - affine.go : 완전연결 계층입니다.
    - binaryCrossEntropy.go : Binary Cross Entropy입니다.
    - layer.go : 레이어의 공통 요소를 인터페이스로 정의하였습니다.
    - relu.go : relu입니다.
    - sigmoid.go : sigmoid입니다.
# AlphaCodium으로 코드 생성: 프롬프트 엔지니어링에서 흐름 엔지니어링으로

[논문](https://arxiv.org/abs/2401.08500) |
[데이터세트](https://huggingface.co/datasets/talrid/CodeContests_valid_and_test_AlphaCodium/blob/main/codecontests_valid_and_test_processed_alpha_codium.zip)

공식 구현
> 탈 리드닉, 데디 크레도, 이타마르 프리드먼 <br/> CodiumAI

## 목차
- [초록](#abstract)
- [설치](#설치)
- [실행 방법](#how-to-run)
- [기술 Q&A](#technical-qa)
- [폭넓은 적용 가능성](#broader-applicability)
- [감사](#감사)
- [인용](#citation)

## 초록

코드 생성 문제는 일반적인 자연어 문제와는 달리 대상 언어의 정확한 구문을 일치시키고, 행복한 경로와 에지 케이스를 식별하고, 문제 사양의 수많은 작은 세부 사항에 주의를 기울이고, 기타 코드별 문제와 요구 사항을 해결해야 합니다. 따라서 자연어 생성에 성공한 많은 최적화 및 요령이 코드 작업에는 효과적이지 않을 수 있습니다.

이 연구에서는 코드 문제에 대한 LLM의 성능을 개선하는 테스트 기반의 다단계 코드 지향 반복 흐름인 AlphaCodium이라는 새로운 접근 방식을 제안합니다.

코드포스와 같은 플랫폼의 경쟁 프로그래밍 문제가 포함된 CodeContests라는 까다로운 코드 생성 데이터 세트에서 AlphaCodium을 테스트했습니다. 제안된 흐름은 일관되게 결과를 크게 개선했습니다.
예를 들어, 유효성 검사 세트에서 GPT-4 정확도(통과율@5)는 잘 설계된 단일 직접 프롬프트의 경우 19%에서 AlphaCodium 플로우를 사용하면 44%로 증가했습니다.

이 작업에서 얻은 많은 원칙과 모범 사례는 일반적인 코드 생성 작업에도 광범위하게 적용될 수 있다고 생각합니다.

<p align="center">
 <table class="tg">
  <tr>
    <td class="tg-c3ow"><img src="./pics/proposed_flow.png" align="center" width="600""></td>
<tr>
    <td class="tg-c3ow"><img src="./pics/iterations.png" align="center" width="600"></td>

  </tr>
</table>
</p>


## 설치

(1) 가상 환경을 설정하고 실행합니다: `pip install -r requirements.txt`를 실행합니다.

(2) `alpha_codium/settings/.secrets_template.toml` 파일을 복제하고 이름을 `.secrets.toml`로 변경한 후 openai API 키를 채웁니다:
```
[openai]
key = "..."
```

(3) 처리된 코드 콘테스트 검증 및 테스트 데이터셋을 [hugging face](https://huggingface.co/datasets/talrid/CodeContests_valid_and_test_AlphaCodium/blob/main/codecontests_valid_and_test_processed_alpha_codium.zip)에서 다운로드하여 압축을 푼 후, 압축을 푼 폴더를 프로젝트의 루트에 넣습니다.

## 실행 방법

### 설정
`alpha_codium/settings/configuration.toml` 파일에는 프로젝트에 대한 설정이 포함되어 있습니다.
`config` 섹션에서 사용하려는 모델("gpt-4", "gpt-3.5-turbo-16k" 등)을 선택할 수 있습니다.

### 특정 문제 해결
AlphaCodium의 특정 문제를 해결하려면 루트 폴더에서
```
python -m alpha_codium.solve_problem \.
--dataset_name /path/to/dataset \.
--split_name test \
--problem_number 0
```
- `dataset_name`은 설치 단계에서 다운로드한 데이터셋 폴더의 경로입니다.
- 유효성 검사 집합에는 117개의 문제가 포함되어 있고 테스트 집합에는 165개의 문제가 포함되어 있으므로 `problem_number` 매개 변수는 그에 따라 (0을 기준으로) 지정해야 합니다.
- `split_name`은 `valid` 또는 `test`일 수 있습니다.
- 구성 파일의 다음 섹션은 다음과 같습니다:
`solve`, `self_reflection`, `possible_solutions`, `generate_ai_tests`, `initial_code_generation`, `public_tests`, `ai_tests`
를 사용하여 흐름의 여러 단계에 대해 가능한 구성을 조정할 수 있습니다.
- 각 실행은 결과를 `alpha_codium/example.log`라는 파일에 기록합니다. 로그 파일을 검토하면 흐름의 각 단계에서 무슨 일이 일어나고 있는지 파악할 수 있습니다.

예제 문제(테스트 세트, 문제 번호 12):
<p align="center">
 <table class="tg">
  <tr>
    <td class="tg-c3ow"><img src="./pics/example_problem.png" align="center" width="600""></td>
    </tr>
</table>
</p>

### 전체 데이터 집합 풀기
AlphaCodium으로 전체 데이터 집합을 풀려면 루트 폴더에서 다음을 실행합니다.:
```
python -m alpha_codium.solve_dataset \
--dataset_name /path/to/dataset \
--split_name test
--database_solution_path /path/to/output/dir/dataset_output.jso
```

- `split_name`은 `valid` 또는 `test`일 수 있습니다.
- `database_solution_path`는 솔루션이 저장될 디렉터리 경로입니다.
- 구성 파일의 `dataset` 섹션에는 데이터셋의 실행 및 평가를 위한 구성이 포함되어 있습니다.
- 이 과정은 시간이 오래 걸리며, 큰 모델(예: GPT-4)과 문제당 여러 번의 반복이 필요한 경우 완료하는 데 며칠이 걸릴 수 있습니다.
- `dataset.num_iterations`는 각 문제에 대한 반복 횟수를 정의합니다(pass@K). 반복 횟수가 많은 경우 최상의 결과를 얻으려면 각 반복에 대해 약간의 무작위성과 다양한 옵션을 도입하는 것이 좋습니다.

### 평가 실행하기

전체 데이터 집합(유효 또는 테스트)에 대한 솔루션을 생성한 후에는 실행하여 평가할 수 있습니다:
```
python -m alpha_codium.evaluate_dataset\
--dataset_name /path/to/dataset\
--split_name test\
--database_solution_path /path/to/output/dir/dataset_output.json
```

## 기술 Q&A
이 프로젝트에 대해 받은 몇 가지 기술적인 질문을 정리했습니다:
___
**Q: '프롬프트 엔지니어링'과 '플로우 엔지니어링'을 비교했을 때 '프롬프트 엔지니어링'에 얼마나 많은 시간을 할애했나요**<br><br>

**A:** 구조화된 출력물은 단순한 프롬프트 엔지니어링의 필요성을 거의 완전히 제거합니다.
우리는 약 95%의 시간을 더 높은 수준의 설계, 추론, 올바른 위치에 데이터 삽입, 즉 "흐름 엔지니어링"에 할애한 것으로 추정합니다.
___

**Q: 데이터 유출이 없었다는 것을 어떻게 알 수 있나요?**

**A:** CodeContests 데이터 세트의 테스트 세트는 2021년 9월 이후에 발표된 문제로 구성되어 있으며, 우리가 사용한 GPT-4 모델 변형(gpt-4-0613)의 데이터 컷오프는 2021년 9월입니다. 따라서 테스트 세트에서 GPT4에 대한 데이터 유출은 없습니다.
DeepSeek와 같은 다른 모델의 경우 확실하지 않습니다. 그러나 [주요 결과](./pics/comparison.png)는 "직접 프롬프트"와 "알파코듐 흐름"을 비교한 것입니다. 데이터 유출은 두 접근 방식 모두에 도움이 될 수 있으므로 AlphaCodium 흐름의 상대적 개선은 여전히 유효합니다.
___

**Q: 이 프로젝트는 특정 프로그래밍 언어와만 관련이 있나요?**

**A:** 아니요. 제안된 흐름은 언어에 구애받지 않습니다. 저희는 Python으로 솔루션을 생성했지만 이 흐름은 모든 언어에 적용할 수 있습니다.
___

**Q: 컨텍스트 창은 어떻게 관리했나요?**

**A:** 저희는 컨텍스트 창이 8192개인 모델을 사용했으며, 이 정도로는 충분하지 않은 경우가 발생하지 않았습니다.
그러나 실제로 사용한 컨텍스트가 커질수록(예: 4000 토큰 이상) 모델이 컨텍스트의 일부 정보를 "무시"하기 시작하는 것을 분명히 관찰했습니다. 따라서 분명한 트레이드오프가 존재합니다:
- 이전 단계의 결과를 컨텍스트에 주입하면 모델이 더 나은 코드를 생성하는 데 도움이 될 수 있습니다.
- 그러나 모델이 문제 설명의 특정 세부 사항과 뉘앙스를 무시하게 될 수도 있습니다.
___

**Q: 이 작업은 LLM 호출 횟수 측면에서 "현실적"인가요?**

**A:** 알파코드와 비교했을 때, 우리는 4배(!) 더 적은 [호출](./pics/computational_effort.png)을 수행합니다(솔루션당 알파코디움은 15~20개의 호출을 수행).
하지만 일부 애플리케이션의 경우 이 수치가 여전히 너무 많을 수 있으며 더 많은 최적화가 필요하다는 것을 인정합니다. 하지만 이 작업에서 얻은 많은 아이디어와 원칙은 호출 횟수가 더 제한되더라도 광범위하게 적용할 수 있다고 생각합니다.

## 광범위한 적용 가능성
이 작업은 CodeContests 데이터 세트에 대한 결과를 제시하지만, 더 광범위하게 적용될 수 있다고 생각합니다.

무엇보다도, 제안된 AlphaCodium [flow](./pics/proposed_flow.png)를 적절히 조정하면 다른 코드 생성 작업을 위한 보다 일반적인 프레임워크로 사용할 수 있다고 생각합니다.

둘째, 이 작업에서 얻은 많은 디자인 개념, 원칙, 요령은 일반적인 코드 생성 작업에 그대로 적용할 수 있습니다. 예를 들어
- **YAML 구조화된 출력**: 모델에 주어진 Pydantic 클래스에 해당하는 YAML 형식의 출력을 생성하도록 요청하기
- **글머리 기호 분석을 통한 의미론적 추론**: 글머리 기호 분석을 통해 문제에 대한 심층적인 이해를 유도하고, 모델이 출력을 논리적인 의미론적 섹션으로 나누도록 하여 결과를 개선합니다.
- **모듈식 코드를 생성할 때 더 효과적**: 모델에 다음과 같이 명확하게 요청할 때 더 효과적입니다: '생성된 코드를 의미 있는 이름과 기능을 가진 작은 하위 기능으로 나누라'고 명확하게 요청하면 버그가 적고 반복적인 수정 단계의 성공률이 높아지는 등 코드가 더 잘 생성되는 것을 관찰할 수 있습니다.
- **이중 검증을 통한 신중한 의사 결정**: 이중 검증 프로세스를 통해 생성된 출력이 주어지면 모델에 동일한 출력을 다시 생성하되 필요한 경우 수정하도록 요청하는 추가 단계를 추가합니다.
- **탐색의 여지 남겨두기**: 모델이 틀릴 수 있으므로 되돌릴 수 없는 결정을 피하고 다양한 가능한 솔루션으로 탐색 및 코드 반복을 할 수 있는 여지를 남겨두는 것이 좋습니다.

위의 목록은 일부입니다. 자세한 내용은 [논문](https://arxiv.org/abs/2401.08500)을 참조하세요. 이 리포지토리에 제공된 코드(./alpha_codium/settings)는 제안된 개념을 더 잘 이해하고 다른 코드 생성 작업에 적용하기 위한 참고 자료로 사용할 수 있습니다.

## 감사
저희의 프로세스 CodeContests 데이터 세트는 원본 [CodeContests](https://huggingface.co/datasets/deepmind/code_contests) 데이터 세트를 기반으로 합니다.
우리 작업과 관련이 없는 훈련 집합을 제거하고 검증 및 테스트 집합에 대한 후처리와 정리를 수행했습니다.


## 인용
```
@misc{ridnik2024code,
      title={알파나트륨으로 코드 생성: 프롬프트 엔지니어링에서 흐름 엔지니어링으로},
      저자={{탈 리드닉과 데디 크레도, 이타마르 프리드먼},
      년도={2024},
      eprint={2401.08500},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

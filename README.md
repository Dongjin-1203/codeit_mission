# codeit_mission
---
## 스프린트 미션1
### 미션 소개
파이썬 응용하기, 객체와 클래스 토픽에서 배운 내용을 바탕으로 예제 문제를 풀어 보세요.

### 스프린트 미션 소개
다음 ipynb 파일을 다운로드 받아, 각 문제를 해결해봅시다.
기초 9문항, 응용 7문항으로 이루어져있어요.
심화 문제는 다소 난이도가 있는 문제라 채점에 포함되지 않으니, 편한 마음으로 도전해봅시다.

### 미션 결과물 제출
구글 드라이브의 결과물 제출 폴더 > `미션01` 폴더에 Colab Notebook을 제출해주세요.
제출 시 파일명은 `미션1_{팀명}_{성함}.ipynb`으로 기재 부탁드립니다.

- 예시: 미션1_1팀_코드잇.ipynb

---
## 스프린트 미션2
### 미션 소개
앞선 토픽에서 배운 내용을 바탕으로, 미션을 수행 해 봅시다.
스프린트 미션 유형에 대한 소개와 제출 방식은 스프린트 미션 안내 및 제출 가이드 참고 부탁드립니다 😊

### 스프린트 미션 소개
이번 미션에서는 Hotel Booking Demand Datasets를 분석해볼 예정입니다.

해당 데이터는 2015.07.01부터 2017.08.31까지의 Resort Hotel과 City Hotel의 예약 데이터를 포함하고 있습니다.

예약 취소는 호텔의 매출에 영향을 끼치는 요소 중 하나인데요, 지금부터 여러분이 City Hotel과 Resort Hotel의 관리인이라고 상상해 보세요. 호텔 관리인으로서 예약 취소와 관련이 있는 요소들을 파악해보고, 예약 취소율을 줄이기 위한 아이디어도 생각해 볼 겁니다. 어떤 조건에서 예약 취소가 빈번하게 발생하는지, 예약 취소와 관련이 있는 요소들이 무엇인지 파악해보고, 마지막으로 어떻게 하면 예약 취소율을 개선할 수 있을지 아이디어도 얻어 보세요!

#### 데이터 관련 추가 설명
해당 데이터는 캐글의 Hotel Booking Demand Datasets(링크)를 가져와서 코드잇에서 일부 수정을 가한 데이터입니다. 따라서 캐글에서 제공하는 데이터와 100% 동일하지는 않습니다.

다른 사람들은 이 데이터를 어떻게 분석하였는지 궁금하다면, 캐글에서 다른 사용자 분들이 분석한 내용들을 참고하는 것 또한 추천드립니다.

추가적으로 캐글에 올라와 있는 데이터 또한 Hotel Booking Demand Datasets라는 아티클에서 가져온 데이터입니다.([아티클 링크](https://www.sciencedirect.com/science/article/pii/S2352340918315191))

해당 아티클은 영문으로 제공되는데요, 데이터가 어떤 방식으로 얻어졌고, 데이터 각 컬럼의 분포 및 경향성에 대해 추가적인 정보를 얻고자 하시는 분들은 해당 아티클을 참고해보는 것 또한 추천드립니다.

#### Kaggle에 올라와 있는 Notebook 참고하는 법
Kaggle에서 다른 사용자 분들이 해당 데이터에 대해 분석한 Notebook은 Code 페이지(링크)에서 확인할 수 있고, Most Votes를 얻은 Dataset Notebooks를 참고하면 여러 명의 캐글 사용자들이 괜찮다고 생각하는 분석 노트북들이 무엇인지 알 수 있습니다.

다만 해당 내용을 무조건적으로 따라하거나 코드를 사용하는 것보다는, 다른 사람들은 이 데이터를 이런 식으로 접근하였구나, 정도로 참고만 하는 것을 추천드립니다! 예약 취소율을 개선하기 위한 본인만의 분석을 진행해 보아요 🙂

### 데이터 설명
- 해당 데이터는 2015.07.01부터 2017.08.31까지의 Resort Hotel과 City Hotel의 예약 데이터를 포함하고 있습니다.
- [데이터 셋 다운로드](https://bakey-api.codeit.kr/api/files/resource?root=static&seqId=11679&version=1&directory=/hotel_data_modified.csv&name=hotel_data_modified.csv)

|컬럼명|설명|
|-----|-----|
|hotel|호텔명 (Resort Hotel 혹은 City Hotel)|
|is_canceled|호텔 예약이 취소되었는지(1) 혹은 취소되지 않았는지(0)를 나타내는 값|
|lead_time|호텔 예약 시점부터 고객의 호텔 도착 시점까지의 기간 (단위: 날짜)|
|arrival_date_year|고객의 호텔 도착 연도|
|arrival_date_month|고객의 호텔 도착 월|
|arrival_date_week_number|고객의 호텔 도착 주|
||(예시: 2015년도 셋째 주에 도착 → arrival_date_week_number = 3)|
|arrival_date_day_of_month|고객의 호텔 도착 일|
||(예시: 3월 2일에 도착 → arrival_date_day_of_month = 2)|
|stays_in_weekend_nights|고객이 호텔에 숙박했거나 예약한 주말 밤 수(토요일~일요일)|
||(예시: 평일 3일 주말 2일 예약한 경우 → stays_in_weekend_nights = 2)|
|stays_in_week_nights|고객이 호텔에 숙박했거나 예약한 주중 밤 수(월요일~금요일)|
||(예시: 평일 3일 주말 2일 예약한 경우 → stays_in_week_nights = 3)|
|adults|예약된 어른의 수|
|children|예약된 어린이의 수|
|babies|예약된 아기의 수|
|meal|예약된 식사 유형|
||- Undefined/SC: 식사 포함되지 않은 옵션|
||- BB: Bed & Breakfast 옵션|
||- HB: Half board (아침 식사 및 추가 식사 1회 - 일반적으로 저녁 식사) 옵션|
||- FB: Full board (아침, 점심, 저녁)|
|country|투숙객의 출신 국가. 카테고리는 ISO 3155-3:2013 형식으로 표시|
||(국가 표기 코드는 링크 참조)|
|market_segment|시장 세그먼트. "TA"는 "Travel Agent", "TO"는 "Tour Operators"를 의미.|
|distribution_channel|예약 유통 채널. "TA"는 "Travel Agent", "TO"는 "Tour Operators"를 의미.|
|is_repeated_guest|이전에 방문을 하였던 손님인지(1) 아닌지(0)를 나타나는 값|
|previous_cancellations|현재 예약 이전에 고객이 취소한 이전 예약 수|
|previous_bookings_not_canceled|현재 예약 이전에 고객이 취소하지 않은 이전 예약 수|
|reserved_room_type|예약한 룸 타입 코드|
|assigned_room_type|배정된 룸 타입 코드. 호텔 운영상의 이유(ex. 초과 예약 등) 또는 고객 요청으로 인해 예약한 객실과 다른 객실 유형이 배정되는 경우가 존재.|
|booking_changes|예약 시점부터 예약 취소/체크인 시점까지 에약에 대한 변경/수정 횟수|
|agent|예약을 진행한 여행사의 ID|
|company|예약을 하였거나 예약금을 지불할 책임이 있는 회사 또는 단체의 ID|
|days_in_waiting_list|예약이 확정되기 전까지 해당 예약이 예약 대기자 명단에 있었던 일수|
|required_car_parking_spaces|고객이 요구하는 주차 공간 수|
|total_of_special_requests|고객의 특별 요청 건수 (ex. 트윈 베드, 아기 침대, 고층, 특별한 뷰 등)|
|reservation_status|예약의 마지막 상태, 총 3가지 카테고리로 구성|
||- Canceled: 고객이 예약을 취소함|
||- Check-Out: 고객이 체크인을 하고 체크아웃을 함|
||- No-Show: 노쇼. 고객이 체크인을 하지 않았고 해당 이유를 알 수 없음.|
|reservations_status_date|마지막 예약 상태(reservation_status)가 설정된 날짜.|

### 미션 결과물 제출
구글 드라이브의 결과물 제출 폴더 > `미션02` 폴더에 Colab Notebook을 제출해주세요.
제출 시 파일명은 `미션2_{팀명}_{성함}.ipynb`으로 기재 부탁드립니다.

예시: `미션2_1팀_코드잇.ipynb`

---

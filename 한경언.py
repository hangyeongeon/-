import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기
df = pd.read_excel('서울대기오염_2019.xlsx의 사본.xlsx')

# 컬럼 이름 확인 및 수정
df = df[~df['측정소명'].isin(['전체', '평균'])]  # '전체', '평균' 행 제거
df.rename(columns={'측정소명': '구'}, inplace=True)  # 컬럼 이름 바꾸기

df['날짜'] = pd.to_datetime(df['날짜'])
df['월'] = df['날짜'].dt.month

# 계절 함수 (봄, 여름, 가을, 겨울)
def get_season(month):
    if 3 <= month <= 5: return '봄'
    if 6 <= month <= 8: return '여름'
    if 9 <= month <= 11: return '가을'
    return '겨울'

df['계절'] = df['월'].apply(get_season)

# 4. 연간 미세먼지 평균
avg_pm10 = df['미세먼지'].mean()
print(f"[4] 연간 미세먼지 평균: {avg_pm10:.2f}")

# 5. 미세먼지 최댓값 날짜
max_pm10_row = df.loc[df['미세먼지'].idxmax()]
print(f"\n[5] 미세먼지 최댓값: {max_pm10_row['날짜']} {max_pm10_row['구']} {max_pm10_row['미세먼지']:.2f}")

# 6. 구별 미세먼지 평균 (구별로 묶어서 평균 계산)
gu_avg = df.groupby('구')['미세먼지'].mean().sort_values(ascending=False)
print(f"\n[6] 구별 미세먼지 평균 상위 5개:\n{gu_avg.head(5)}")

# 7. 계절별 미세먼지/초미세먼지 평균 (계절별로 묶어서 평균 계산)
season_avg = df.groupby('계절')[['미세먼지', '초미세먼지']].mean()
print(f"\n[7] 계절별 미세먼지/초미세먼지 평균:\n{season_avg}")

# 8. 미세먼지 등급 분류 (좋음, 보통, 나쁨, 매우나쁨)
def pm_grade(v):
    if v <= 30: return '좋음'
    if v <= 80: return '보통'
    if v <= 150: return '나쁨'
    return '매우나쁨'

df['등급'] = df['미세먼지'].apply(pm_grade)
grade_counts = df['등급'].value_counts(normalize=True).mul(100).round(2)
print(f"\n[8] 미세먼지 등급별 비율:\n{grade_counts}")

# 9. 구별 '좋음' 등급 비율 상위 5개
good_ratio = df[df['등급'] == '좋음'].groupby('구').size() / df.groupby('구').size() * 100
good_ratio = good_ratio.sort_values(ascending=False).head(5).round(2)
print(f"\n[9] 구별 '좋음' 등급 비율 상위 5개:\n{good_ratio}")

# 10. 1년간 일별 미세먼지 추이 그래프
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='날짜', y='미세먼지')
plt.title('2019년 서울시 일별 미세먼지 추이')
plt.xlabel('날짜')
plt.ylabel('미세먼지 (㎍/㎥)')
plt.tight_layout()
plt.show()

# 11. 계절별 미세먼지 등급 비율 그래프
season_grade = df.groupby(['계절', '등급']).size().unstack().fillna(0)  # 빈 값 0으로 채우기
season_grade_pct = season_grade.div(season_grade.sum(axis=1), axis=0) * 100
plt.figure(figsize=(10, 6))
season_grade_pct.plot(kind='bar', stacked=True)
plt.title('2019년 서울시 계절별 미세먼지 등급 분포')
plt.xlabel('계절')
plt.ylabel('비율 (%)')
plt.legend(title='등급')
plt.tight_layout()
plt.show()

df.to_csv('card_output.csv', index=False, encoding='utf-8-sig')

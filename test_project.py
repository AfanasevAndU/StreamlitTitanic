import pytest
import pandas as pd
from io import StringIO
from titanic import load_data, filter_survived_women, group_by_class  

SAMPLE_DATA = """
PassengerId,Survived,Pclass,Sex,Age,Fare
1,1,1,female,29,100
2,0,2,male,35,20
3,1,3,female,22,10
4,1,1,female,40,150
5,0,3,male,30,5
"""

@pytest.fixture
def sample_df():
    return pd.read_csv(StringIO(SAMPLE_DATA))

def test_load_data():
    df = load_data()
    assert not df.empty
    assert 'Survived' in df.columns

def test_filter_survived_women(sample_df):
    filtered_df = filter_survived_women(sample_df)
    assert len(filtered_df) == 3
    assert all(filtered_df['Sex'] == 'female')
    assert all(filtered_df['Survived'] == 1)

def test_group_by_class(sample_df):
    filtered_df = filter_survived_women(sample_df)
    grouped_df = group_by_class(filtered_df)
    assert len(grouped_df) == 2
    assert set(grouped_df['Pclass']) == {1, 3}
    assert grouped_df.loc[grouped_df['Pclass'] == 1, 'Count'].values[0] == 2
    assert grouped_df.loc[grouped_df['Pclass'] == 1, 'Fare_Range_Min'].values[0] == 100
    assert grouped_df.loc[grouped_df['Pclass'] == 1, 'Fare_Range_Max'].values[0] == 150

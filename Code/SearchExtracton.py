from init import *
from allURL import *

def Read_CSV(completeName, className):
    df = pd.read_csv(completeName)
    falseSearch = (df[className]==1).value_counts()[0]
    trueSearch = (df[className]==1).value_counts()[1]
    total = len(df[className])
    print(className.split('Class')[0])
    print("================================")
    print('Accuracy: {:.1f}%'.format(((total-falseSearch)/total)*100))
    print('Number of False result: {:.1f}'.format(falseSearch))
    print('Number of True result: {:.1f}'.format(trueSearch))
    print('Number of Total result: {:.1f}'.format(total))
def main():
    save_path = '/Users/mdjavedulferdous/Desktop/TiiS/Code/'
    search_name = "search.csv"
    page_name = 'page.csv'
    sort_name = 'sort.csv'
    filter_name = 'filter.csv'
    searchName = os.path.join(save_path, search_name)
    pageName = os.path.join(save_path, page_name)
    sortName = os.path.join(save_path, sort_name)
    filterName = os.path.join(save_path, filter_name)
    Read_CSV(searchName,'sClass')
    Read_CSV(pageName,'pageClass')
    Read_CSV(sortName,'sortClass')
    Read_CSV(filterName,'filterClass')
if __name__ == "__main__":
    main()
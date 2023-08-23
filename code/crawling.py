### 다음 뉴스 크롤링 ###
def crawler_daum():
    
    ### 날짜 설정 ###
    start_date = datetime.date(2017, 1, 1)  # 시작 날짜
    end_date = datetime.date(2023, 4, 25)  # 끝 날짜
    delta = datetime.timedelta(days=1)  # 날짜 증가량
    
    title_list = []
    source_list = []
    name_list = []
    date_list = []
    contents_list = []
    link_list = []
    
    while start_date <= end_date:
        number = int(start_date.strftime('%Y%m%d'))
        start_date += delta
        print('daum', number, '날짜')
        for r_page in range(1, 100):
            url_root = "https://news.daum.net/breakingnews/politics/dipdefen?page="
            url = urlopen(url_root + str(r_page) + "&regDate=" + str(number))
            soup = BeautifulSoup(url.read(), 'html.parser') 
            
            # if문용 임시 변수
            url2 = urlopen(url_root + str(r_page + 1) + "&regDate=" + str(number))
            temp1 = BeautifulSoup(url2.read(), 'html.parser')
            temp2 = soup.find('em', 'num_page').text
            temp3 = temp1.find('em', 'num_page').text

            if temp2 != temp3:
                links = soup.find_all('strong', 'tit_thumb')
                for link in links[0:15]:
                    article_url = link.find('a').attrs['href']
                    link_list.append(article_url)
                    m_soup = BeautifulSoup(urlopen(article_url).read(), 'html.parser')
                    
                    # 기사 제목 추출
                    try:
                        title_lists = m_soup.find('h3', 'tit_view')
                        title_list.append(title_lists.text)
                    except AttributeError:
                        title_list.append("None")      

                    # 신문사 추출
                    try:
                        source_lists = m_soup.find('h1', 'doc-title')
                        source_list.append(source_lists.find('a', {'id': 'kakaoServiceLogo'}).text.strip())
                    except AttributeError:
                        source_list.append("None")             
                        
                    # 기자 추출
                    try:
                        name_lists = m_soup.find('span', 'txt_info')
                        name_list.append(name_lists.text)
                    except AttributeError:
                        name_list.append("None")                           

                    # 날짜 추출
                    try:
                        date_lists = m_soup.find('span', 'num_date').text
                        date_list.append(date_lists.split(":")[0][:-3])
                    except AttributeError:
                        date_list.append("None")      

                    # 본문 추출
                    try:
                        contents_lists = m_soup.find('div', 'article_view')
                        contents_list.append(contents_lists.find('section').text.strip())
                    except AttributeError:
                        contents_list.append("None")                           
                        
            else:
                break
            print('daum', r_page, '페이지')

   
        # 마지막 페이지 추가            
        url_root = "https://news.daum.net/breakingnews/politics/dipdefen?page=3000&regDate="
        url = urlopen(url_root + str(number))
        test = soup.find('ul', 'list_news2 list_allnews')
        test2 = str(test.find_all('li'))
        count = int(test2.count("<li>"))
        soup = BeautifulSoup(url.read(), 'html.parser')
        links = soup.find_all('strong', 'tit_thumb')
        for link in links[0 : count]:
            article_url = link.find('a').attrs['href']
            link_list.append(article_url)
            m_soup = BeautifulSoup(urlopen(article_url).read(), 'html.parser')

            # 기사 제목 추출
            try:
                title_lists = m_soup.find('h3', 'tit_view')
                title_list.append(title_lists.text)
            except AttributeError:
                title_list.append("None") 

            # 신문사 추출
            try:
                source_lists = m_soup.find('h1', 'doc-title')
                source_list.append(source_lists.find('a', {'id': 'kakaoServiceLogo'}).text.strip())
            except AttributeError:
                source_list.append("None") 
                
            # 기자 추출
            try:
                name_lists = m_soup.find('span', 'txt_info')
                name_list.append(name_lists.text)
            except AttributeError:
                name_list.append("None")                   

            # 날짜 추출
            try:
                date_lists = m_soup.find('span', 'num_date').text
                date_list.append(date_lists.split(":")[0][:-3])
            except AttributeError:
                date_list.append("None") 

            # 본문 추출
            try:
                contents_lists = m_soup.find('div', 'article_view')
                contents_list.append(contents_lists.find('section').text.strip())   
            except AttributeError:
                contents_list.append("None") 
    
    # 리스트 딕셔너리로 저장
    result = {'title' : title_list, 'source' : source_list, 'name' : name_list, 'date' : date_list, 'contents' : contents_list, 'link' : link_list}
    news_df = pd.DataFrame(result)
    
    # excel 저장
    news_df.to_excel("./daum_news_crawling.xlsx", sheet_name = 'daum_news')

if __name__ == '__main__':
    crawler_daum()

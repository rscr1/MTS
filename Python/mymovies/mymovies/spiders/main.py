import scrapy
import csv

class FilmSpider(scrapy.Spider):
    name = 'film_spider'
    start_urls = ['https://ru.wikipedia.org/wiki/%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%A4%D0%B8%D0%BB%D1%8C%D0%BC%D1%8B_%D0%BF%D0%BE_%D0%B0%D0%BB%D1%84%D0%B0%D0%B2%D0%B8%D1%82%D1%83']  # Замените на свой URL с фильмами

    def parse(self, response):
        for film in response.css('div.film'):
            title = film.css('h2::text').get()
            genre = film.css('span.genre::text').get()
            director = film.css('span.director::text').get()
            country = film.css('span.country::text').get()
            year = film.css('span.year::text').get()

            with open('movies.csv', 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Title', 'Genre', 'Director', 'Country', 'Year', 'IMDB Rating']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if csvfile.tell() == 0:
                    writer.writeheader()

                writer.writerow({
                    'Title': title,
                    'Genre': genre,
                    'Director': director,
                    'Country': country,
                    'Year': year,
                    'IMDB Rating': None
                })

                imdb_url = film.css('a.imdb-link::attr(href)').get()
                if imdb_url:
                    yield scrapy.Request(imdb_url, callback=self.parse_imdb_rating, meta={'title': title})

    def parse_imdb_rating(self, response):
        title = response.meta['title']
        imdb_rating = response.css('div.ratingValue strong span::text').get()

        with open('movies.csv', 'r', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Title', 'Genre', 'Director', 'Country', 'Year', 'IMDB Rating']
            reader = csv.DictReader(csvfile)

            rows = list(reader)
            for row in rows:
                if row['Title'] == title:
                    row['IMDB Rating'] = imdb_rating

        with open('movies.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

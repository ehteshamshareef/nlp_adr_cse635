import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { delay, map } from 'rxjs/operators';

const TOTAL_PAGES = 7;

export class NewsPost {
  tweet_id: Number;
  begin: string;
  end: string;
  type: string;
  extraction : string;
  drug : string;
  tweet: string;
  meddra_code : Number;
  meddra_term : string
}

@Injectable()
export class NewsService {

  constructor(private http: HttpClient) {}

  load(page: number, pageSize: number): Observable<NewsPost[]> {
    const startIndex = ((page - 1) % TOTAL_PAGES) * pageSize;

    return this.http
      .get<NewsPost[]>('http://0.0.0.0:80/getTweets')
      .pipe(
        map(news => news.splice(startIndex, pageSize)),
        delay(1500),
      );
  }
}

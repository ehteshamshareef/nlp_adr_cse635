import { Injectable } from '@angular/core';
import { HttpClient } from "@angular/common/http"
import { Phrase } from "./phrase"
import { Observable } from "rxjs/Observable"
import { HttpParams } from "@angular/common/http";

@Injectable({
  providedIn: 'root'
})
export class PhraseService{
    private _url : string = "0.0.0.0:80/getTweets"
    constructor(private http: HttpClient){ }
    
    getData(phrase) : Observable<Phrase[]>{
        let params = new HttpParams();
        params = params.append('phrase', phrase);
        return this.http.get<Phrase[]>("http://localhost:80/getTweets",{params: params});
    }
}

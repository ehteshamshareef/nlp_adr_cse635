import { Component } from '@angular/core';
import {PhraseService} from "../phrase.service"

@Component({
  selector: 'ngx-form-layouts',
  styleUrls: ['./form-layouts.component.scss'],
  templateUrl: './form-layouts.component.html',
})
export class FormLayoutsComponent {
  public receivedData : any = {}
  public flag : boolean = false
  constructor(private _formservice: PhraseService){}

  clickFunction(searchValue : String){
    this.flag = true
    this._formservice.getData(searchValue)
      .subscribe(data => {
        this.receivedData = data;
        console.log('data', data);
     })  
  }
}

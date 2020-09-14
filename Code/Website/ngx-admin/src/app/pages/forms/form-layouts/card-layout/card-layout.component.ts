import { Component, Input, OnInit } from '@angular/core';
import { Phrase } from '../../phrase';
@Component({
  selector: 'ngx-card-layout',
  templateUrl: './card-layout.component.html',
  styleUrls: ['./card-layout.component.scss']
})
export class CardLayoutComponent implements OnInit {

  @Input() post: Phrase;
  constructor() { }
  ngOnInit(): void {
  }

  

}

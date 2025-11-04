import { Component, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import {TaskList} from './task-list/task-list'
import { EventAnalysisComponent } from './event-analysis/event-analysis.component';
import { CompactEventCardComponent } from './compact-event-card/compact-event-card.component';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, TaskList, EventAnalysisComponent, CompactEventCardComponent],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  protected readonly title = signal('hello-angular');
}

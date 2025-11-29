import { CommonModule, DecimalPipe } from '@angular/common';
import { Component, Input } from '@angular/core';

export interface SimulationResult {
  cashpointId: string;
  cashpointName: string;
  location: string;

  originalServiceDays: string;
  simulatedServiceDays: string;

  savingsPercentage: number;   // e.g. 14.28%
  stockoutRisk: 'Low' | 'Medium' | 'High';

  recommendation: string;
}

@Component({
  selector: 'app-simulation-results',
  templateUrl: './simulation-results.component.html',
  standalone: true,
  imports: [CommonModule, DecimalPipe]
})
export class SimulationResultsComponent {
  @Input() results: SimulationResult[] = [];

  constructor() {}

  trackById(_: number, item: SimulationResult) {
    return item.cashpointId;
  }
}

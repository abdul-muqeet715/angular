import { Component, signal } from '@angular/core';
import { MachineListComponent } from '../machine-list/machine-list.component';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';

@Component({
  selector: 'app-event-analysis',
  imports: [CommonModule, MachineListComponent],
  templateUrl: './event-analysis.component.html',
  styleUrl: './event-analysis.component.css',
})
export class EventAnalysisComponent {
  constructor(private router: Router) {}

  aiInsight = signal('');
  lastGenerated = signal('');

  eventData = {
    '1': {
      name: 'Martin Luther King Day',
      type: 'Holiday',
      date: '2024-01-15',
      impact: 'high',
      affectedCashpoints: 23,
      amountChange: 1800000,
      description: 'Public holiday leading to increased cash demand across retail and tourism areas.',
      cashpoints: [
        {
          machine_name: 'ATM-Mall-Central',
          crncy_id: 'USD',
          date_: '2024-01-15',
          forecast_amount: 45000,
          actual_amount: 58000,
          diff: 13000,
          location: 'Shopping Center'
        },
        {
          machine_name: 'ATM-Downtown-01',
          crncy_id: 'USD',
          date_: '2024-01-15',
          forecast_amount: 32000,
          actual_amount: 41000,
          diff: 9000,
          location: 'Main Street Branch'
        },
        {
          machine_name: 'ATM-Tourist-Area',
          crncy_id: 'USD',
          date_: '2024-01-15',
          forecast_amount: 28000,
          actual_amount: 35000,
          diff: 7000,
          location: 'Historic District'
        }
      ]
    },
    '2': {
      name: 'Downtown Festival',
      type: 'Local Event',
      date: '2024-01-16',
      impact: 'medium',
      affectedCashpoints: 8,
      amountChange: 500000,
      description: 'Local festival increasing foot traffic and cash demand in downtown area.',
      cashpoints: [
        {
          machine_name: 'ATM-Festival-Main',
          crncy_id: 'USD',
          date_: '2024-01-16',
          forecast_amount: 25000,
          actual_amount: 31000,
          diff: 6000,
          location: 'Festival Grounds'
        },
        {
          machine_name: 'ATM-Downtown-02',
          crncy_id: 'USD',
          date_: '2024-01-16',
          forecast_amount: 22000,
          actual_amount: 27000,
          diff: 5000,
          location: 'City Center'
        }
      ]
    }
  };

  selectedId = '1';
  data = this.eventData[this.selectedId as keyof typeof this.eventData];

  generateAIInsight(): void {
    const insightText = [
      `The ${this.data.name} has significantly impacted cash demand patterns across ${this.data.affectedCashpoints} cashpoints. Our analysis shows a positive variance of ${this.data.amountChange.toLocaleString()} compared to baseline forecasts.`,
      `Key observations: ${this.data.type} events typically drive increased cash demand in retail and tourism areas.`,
      `Recommendation: Increase cash replenishment schedules for affected locations by 25% during similar future events.`
    ].join('\n\n');

    this.aiInsight.set(insightText);
    this.lastGenerated.set(new Date().toLocaleString());
  }

  formatCurrency(amount: number): string {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount);
  }

  onEventClick(){
    this.router.navigate(['/']);
  }

}

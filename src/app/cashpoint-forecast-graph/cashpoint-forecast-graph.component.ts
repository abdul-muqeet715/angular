import { Component, Input } from '@angular/core';
import { NgChartsModule } from 'ng2-charts';
import { ChartConfiguration } from 'chart.js';

@Component({
  selector: 'app-cashpoint-forecast-graph',
  standalone: true,
  imports: [NgChartsModule],
  templateUrl: './cashpoint-forecast-graph.component.html',
})
export class CashpointForecastGraphComponent {
  @Input() cashpoints: any[] = [];

  get chartData(): ChartConfiguration<'bar'>['data'] {
    return {
      labels: this.cashpoints.map(c => c.machine_name),
      datasets: [
        {
          label: 'Forecasted',
          data: this.cashpoints.map(c => c.forecast_amount),
          backgroundColor: '#3b82f6',
        },
        {
          label: 'Actual',
          data: this.cashpoints.map(c => c.actual_amount),
          backgroundColor: '#16a34a',
        },
      ],
    };
  }

  chartOptions: ChartConfiguration<'bar'>['options'] = {
    responsive: true,
    plugins: { legend: { position: 'bottom' } },
  };
}

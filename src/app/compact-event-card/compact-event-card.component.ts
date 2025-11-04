import { CommonModule } from '@angular/common';
import { Component, Input, Output, EventEmitter } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-compact-event-card',
  imports: [CommonModule],
  templateUrl: './compact-event-card.component.html',
  styleUrl: './compact-event-card.component.css',
})
export class CompactEventCardComponent {
  constructor(private router: Router) {}

  currentEvents = [
    { id: '1', type: 'Holiday', name: 'Martin Luther King Day', impact: 'high', date: '2024-01-15', affectedCashpoints: 23, amountChange: 1800000, isIncluded: false },
    { id: '2', type: 'Local Event', name: 'Downtown Festival', impact: 'medium', date: '2024-01-16', affectedCashpoints: 8, amountChange: 500000, isIncluded: true },
    { id: '3', type: 'Maintenance', name: 'ATM Network Update', impact: 'low', date: '2024-01-17', affectedCashpoints: 5, amountChange: -200000, isIncluded: false },
  ];

  getImpactColor(impact: string) {
    switch (impact) {
      case 'high': return 'bg-red-100 text-red-700 border-red-200';
      case 'medium': return 'bg-yellow-100 text-yellow-700 border-yellow-200';
      case 'low': return 'bg-green-100 text-green-700 border-green-200';
      default: return 'bg-gray-100 text-gray-700 border-gray-200';
    }
  }

  getIncludeColor(included: boolean){
    if(included) return 'border-green-200 bg-green-50/30';
    return 'border-orange-200 bg-orange-50/30'
  }

  formatCurrency(amount: number): string {
  const absAmount = Math.abs(amount);
  if (absAmount >= 10000000) return `₹${(absAmount / 10000000).toFixed(1)}Cr`;
  if (absAmount >= 100000) return `₹${(absAmount / 100000).toFixed(1)}L`;
  return `₹${(absAmount / 1000).toFixed(0)}K`;
}


  onEventClick(id: string) {
    this.router.navigate(['/event-analysis', id]);
  }
}
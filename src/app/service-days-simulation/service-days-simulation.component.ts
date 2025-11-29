import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { SimulationResultsComponent } from '../simulation-results/simulation-results.component';
import { SimulationSettingsFormComponent } from '../simulation-settings-form/simulation-settings-form.component';
import { SimulationSetting } from '../simulation.model';
import { SimulationResult } from '../simulation.model';
import { Cashpoint } from '../simulation.model';
@Component({
  selector: 'app-service-days-simulation',
  imports: [CommonModule, SimulationResultsComponent, SimulationSettingsFormComponent],
  templateUrl: './service-days-simulation.component.html',
  styleUrl: './service-days-simulation.css',
  standalone: true
})
export class ServiceDaysSimulation {
activeTab: 'settings' | 'results' = 'settings';
  settings: SimulationSetting[] = [];
  showForm = false;
  editingSetting?: SimulationSetting;
  selectedCashpoints: string[] = [];
  selectedSettingId: string | null = null;
  simulationResults: SimulationResult[] = [];

  mockCashpoints: Cashpoint[] = [
    { id: 'CP001', name: 'ATM-Downtown-01', location: 'Main Street Branch' },
    { id: 'CP002', name: 'ATM-Mall-Central', location: 'Shopping Center' },
    { id: 'CP003', name: 'ATM-Airport-T1', location: 'Terminal 1' },
    { id: 'CP004', name: 'ATM-University', location: 'Campus Center' },
    { id: 'CP005', name: 'ATM-Hospital', location: 'Medical District' },
    { id: 'CP006', name: 'ATM-Bank-HQ', location: 'Corporate Headquarters' },
    { id: 'CP007', name: 'ATM-Retail-Park', location: 'Shopping District' },
    { id: 'CP008', name: 'ATM-Gas-Station', location: 'Highway 101' }
  ];

  handleSaveSetting(setting: SimulationSetting) {
    if (this.editingSetting) {
      this.settings = this.settings.map(s => s.id === setting.id ? setting : s);
    } else {
      this.settings.push(setting);
    }

    this.showForm = false;
    this.editingSetting = undefined;
  }

  handleEditSetting(setting: SimulationSetting) {
    this.editingSetting = setting;
    this.showForm = true;
  }

  handleDeleteSetting(id: string) {
    if (confirm('Are you sure you want to delete this setting?')) {
      this.settings = this.settings.filter(s => s.id !== id);
      if (this.selectedSettingId === id) {
        this.selectedSettingId = null;
      }
    }
  }

  toggleCashpoint(id: string) {
    if (this.selectedCashpoints.includes(id)) {
      this.selectedCashpoints = this.selectedCashpoints.filter(c => c !== id);
    } else {
      this.selectedCashpoints.push(id);
    }
  }

  toggleAllCashpoints() {
    if (this.selectedCashpoints.length === this.mockCashpoints.length) {
      this.selectedCashpoints = [];
    } else {
      this.selectedCashpoints = this.mockCashpoints.map(c => c.id);
    }
  }

  runSimulation() {
    if (!this.selectedSettingId || this.selectedCashpoints.length === 0) {
      alert('Select setting and cashpoints');
      return;
    }

    const results: SimulationResult[] = this.selectedCashpoints.map(cpId => {
      const cp = this.mockCashpoints.find(c => c.id === cpId)!;
      const original = Math.floor(Math.random() * 3) + 5;
      const simulated = Math.floor(Math.random() * 2) + 3;
      const cashSaved = Math.floor(Math.random() * 50000) + 10000;
      const savingsPct = ((original - simulated) / original) * 100;

      const risks: ('Low' | 'Medium' | 'High')[] = ['Low', 'Medium', 'High'];

      return {
        cashpointId: cpId,
        cashpointName: cp.name,
        location: cp.location,
        originalServiceDays: "11111",
        simulatedServiceDays: "22222",
        cashSaved,
        savingsPercentage: savingsPct,
        serviceEfficiency: Math.floor(Math.random() * 20) + 80,
        stockoutRisk: risks[Math.floor(Math.random() * risks.length)],
        recommendation: simulated < original
            ? 'Reduce service frequency to optimize costs'
            : 'Maintain current service schedule'
      };
    });

    this.simulationResults = results;
    this.activeTab = 'results';
  }
}

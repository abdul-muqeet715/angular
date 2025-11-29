import { Component, EventEmitter, Input, Output } from '@angular/core';
import { SimulationSetting } from '../simulation.model';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-simulation-settings-form',
  templateUrl: './simulation-settings-form.component.html',
  standalone: true,
  imports : [CommonModule, FormsModule]
})
export class SimulationSettingsFormComponent {

  @Input() setting?: SimulationSetting;
  @Input() isEditing = false;
  @Output() onSave = new EventEmitter<SimulationSetting>();
  @Output() onCancel = new EventEmitter<void>();

  DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

  CYCLES = [
    'Weekly', 'Bi-Weekly', 'Tri-Weekly', 'Forth-Week',
    'Monthly Week-1', 'Monthly Week-2', 'Monthly Week-3', 'Monthly Week-4'
  ];

  formData: SimulationSetting = {
    id: '',
    modelId: '',
    maxHoldingAmount: 0,
    interestRate: 0,
    safetyStock: 0,
    preWithdrawalPct: 0,

    serviceMode: 'ADD',

    addDays: [],
    replaceDays: [],
    returnDays: [],
    unplannedDays: [],

    addCycles: [],
    replaceCycles: [],
    addReplaceCycles: [],
    returnCycles: [],

    plannedAddCost: 0,
    plannedReplaceCost: 0,
    plannedReturnCost: 0,
    plannedCombinedCost: 0,

    unplannedAddCost: 0,
    unplannedReplaceCost: 0,
    unplannedReturnCost: 0
  };

  ngOnInit() {
    if (this.setting) {
      this.formData = { ...this.setting };
    }
  }

  toggleDay(field: keyof SimulationSetting, day: string) {
    const arr = this.formData[field] as string[];
    // this.formData[field] = arr.includes(day)
    //   ? arr.filter(d => d !== day)
    //   : [...arr, day];
  }

  toggleCycle(field: keyof SimulationSetting, cycle: string) {
    const arr = this.formData[field] as string[];
    // this.formData[field] = arr.includes(cycle)
    //   ? arr.filter(c => c !== cycle)
    //   : [...arr, cycle];
  }

  save() {
    if (!this.formData.id) {
      this.formData.id = 'setting_' + Date.now();
    }
    this.onSave.emit(this.formData);
  }
}

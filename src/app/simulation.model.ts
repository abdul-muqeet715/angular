export interface SimulationSetting {
  id: string;
  modelId: string;
  maxHoldingAmount: number;
  interestRate: number;
  safetyStock: number;
  preWithdrawalPct: number;
  serviceMode: 'ADD' | 'REPLACE' | 'ADD_REPLACE';

  addDays: string[];
  replaceDays: string[];
  returnDays: string[];
  unplannedDays: string[];

  addCycles: string[];
  replaceCycles: string[];
  addReplaceCycles: string[];
  returnCycles: string[];

  plannedAddCost?: number;
  plannedReplaceCost?: number;
  plannedReturnCost?: number;
  plannedCombinedCost?: number;

  unplannedAddCost?: number;
  unplannedReplaceCost?: number;
  unplannedReturnCost?: number;
}

export interface Cashpoint {
  id: string;
  name: string;
  location: string;
}

export interface SimulationResult {
  cashpointId: string;
  cashpointName: string;
  location: string;
  originalServiceDays: string;
  simulatedServiceDays: string;
  cashSaved: number;
  savingsPercentage: number;
  serviceEfficiency: number;
  stockoutRisk: 'Low' | 'Medium' | 'High';
  recommendation: string;
}

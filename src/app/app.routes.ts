import { Routes } from '@angular/router';
import { EventAnalysisComponent } from './event-analysis/event-analysis.component';
import { CompactEventCardComponent } from './compact-event-card/compact-event-card.component';
import { ServiceDaysSimulation } from './service-days-simulation/service-days-simulation.component';

export const routes: Routes = [
    { path: '', component: CompactEventCardComponent},
    { path: 'event-analysis/:id', component: EventAnalysisComponent },
    { path: 'simulation', component: ServiceDaysSimulation}
];

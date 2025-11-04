import { Component, Input } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';

@Component({
  selector: 'app-machine-list',
  imports: [CommonModule, DecimalPipe],
  templateUrl: './machine-list.component.html',
  styleUrl: './machine-list.component.css',
})
export class MachineListComponent {
  @Input() machines: any[] = [];
  @Input() showDiff = false;
  @Input() onMachineClick?: (id: string) => void;
}

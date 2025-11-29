import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SimulationSettingsFormComponent } from './simulation-settings-form.component';

describe('SimulationSettingsFormComponent', () => {
  let component: SimulationSettingsFormComponent;
  let fixture: ComponentFixture<SimulationSettingsFormComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SimulationSettingsFormComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(SimulationSettingsFormComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

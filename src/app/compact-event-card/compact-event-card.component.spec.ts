import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CompactEventCard } from './compact-event-card';

describe('CompactEventCard', () => {
  let component: CompactEventCard;
  let fixture: ComponentFixture<CompactEventCard>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CompactEventCard]
    })
    .compileComponents();

    fixture = TestBed.createComponent(CompactEventCard);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
